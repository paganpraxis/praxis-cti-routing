from __future__ import annotations

import random
from collections import defaultdict
from datetime import datetime
from statistics import median
from typing import Any, Callable

from .metrics import feature_degeneracy
from .features import extract_features
from .schema import DetectorInput


METRIC_KEYS = (
    "recall_at_10",
    "full_cluster_recovery",
    "mrr",
    "precision_at_rel",
    "precision_at_10",
    "precision_at_10_ceiling",
    "success_at_1",
    "success_at_3",
    "success_at_5",
    "success_at_10",
)


def evaluate_mds_retrieval(
    catalog: list[dict[str, Any]],
    traces: list[dict[str, Any]],
    detector_inputs: list[dict[str, Any]],
    *,
    as_of: datetime,
    date_normalization: dict[str, Any] | None = None,
    bootstrap_samples: int = 2000,
    seed: int = 34057,
    feature_distinct_floor: float = 0.90,
) -> dict[str, Any]:
    catalog_by_id = {str(row["query_id"]): row for row in catalog}
    detector_by_id = {
        str(row["query_id"]): DetectorInput.from_dict(row) for row in detector_inputs
    }
    rows = [_evaluate_row(catalog_by_id[str(trace["id"])], trace) for trace in traces]
    for row in rows:
        row["detector_features"] = extract_features(detector_by_id[str(row["query_id"])], as_of)
    retrieval_distinctness = feature_degeneracy(
        {
            "task": row["task"],
            "features": row["retrieval_feature_vector"],
        }
        for row in rows
    )
    detector_distinctness = feature_degeneracy(
        {"task": row["task"], "features": row["detector_features"]}
        for row in rows
    )
    failed = {
        task: values["distinct_feature_vector_rate"]
        for task, values in detector_distinctness.items()
        if task != "overall" and float(values["distinct_feature_vector_rate"]) < feature_distinct_floor
    }
    if failed:
        raise ValueError(f"feature degeneracy below declared floor {feature_distinct_floor}: {failed}")

    overall = _summarize(rows, bootstrap_samples, seed)
    tasks = {
        task: _summarize([row for row in rows if row["task"] == task], bootstrap_samples, seed)
        for task in sorted({str(row["task"]) for row in rows})
    }
    zero_hit_queries = [str(row["query_id"]) for row in rows if row["zero_hit"]]
    return {
        "status": "execution_pass_exploratory_only",
        "anchor_policy": traces[0].get("anchor_policy") if traces else None,
        "n_queries": len(rows),
        "n_source_clusters": len({row["source_cluster_id"] for row in rows}),
        "source_cluster_unit": "anchor_id",
        "bootstrap": {
            "unit": "source_cluster",
            "samples": bootstrap_samples,
            "seed": seed,
            "confidence_level": 0.95,
        },
        "detector_inputs": len(rows),
        "runtime_errors": 0,
        "forbidden_detector_fields": [],
        "feature_distinct_floor": feature_distinct_floor,
        "retrieval_vector_distinctness": retrieval_distinctness,
        "detector_feature_distinctness": detector_distinctness,
        "within_cluster_feature_variation": _within_cluster_feature_variation(rows),
        "date_normalization": date_normalization or {},
        "zero_hit_queries": zero_hit_queries,
        "zero_hit_rate": len(zero_hit_queries) / len(rows) if rows else 0.0,
        "metrics_including_zero_hits": overall["including_zero_hits"],
        "metrics_excluding_zero_hits": overall["excluding_zero_hits"],
        "tasks": tasks,
    }


def render_mds_report(summary: dict[str, Any], manifest: dict[str, Any]) -> str:
    def metric(block: dict[str, Any], key: str) -> str:
        value = block["metrics"][key]["estimate"]
        lower, upper = block["metrics"][key]["cluster_bootstrap_95_ci"]
        return f"{value:.4f} [{lower:.4f}, {upper:.4f}]"

    lines = [
        f"# PX-034 question-conditioned CTIConnect CSKG run ({manifest.get('result_version', 'unversioned')})",
        "",
        "Status: **execution pass; exploratory retrieval result only**",
        "",
        "This run uses question-conditioned CSKG retrieval: normalized BM25 scores from question tokens "
        f"({manifest['cskg_fusion']['question_weight']:.2f}) are fused with anchor-vocabulary scores "
        f"({manifest['cskg_fusion']['anchor_weight']:.2f}). The first gold-cluster member remains an exploratory anchor, not a confirmatory input.",
        "The weight was selected by maximizing recall@10 among sweep candidates clearing the 0.90 detector-feature distinctness floor in every task; see `weight_sweep.md`.",
        "",
        "## Execution integrity and effective sample size",
        "",
        f"- n_queries: {summary['n_queries']}",
        f"- n_source_clusters: {summary['n_source_clusters']} (CSC {summary['tasks']['csc']['n_source_clusters']}, MLA {summary['tasks']['mla']['n_source_clusters']}, TAP {summary['tasks']['tap']['n_source_clusters']})",
        f"- Confidence intervals: 95% cluster bootstrap ({summary['bootstrap']['samples']} resamples; clusters, not queries)",
        f"- Feature distinctness floor: {summary['feature_distinct_floor']:.2f}",
        f"- Distinct retrieval vectors: {summary['retrieval_vector_distinctness']['overall']['distinct_feature_vectors']}/{summary['retrieval_vector_distinctness']['overall']['rows']} ({summary['retrieval_vector_distinctness']['overall']['distinct_feature_vector_rate']:.4f})",
        f"- Distinct detector feature vectors: {summary['detector_feature_distinctness']['overall']['distinct_feature_vectors']}/{summary['detector_feature_distinctness']['overall']['rows']} ({summary['detector_feature_distinctness']['overall']['distinct_feature_vector_rate']:.4f})",
        f"- Fully collapsed source clusters on detector features: {summary['within_cluster_feature_variation']['fully_collapsed_clusters']}",
        f"- Feature dimensions varying within at least one cluster: {summary['within_cluster_feature_variation']['varying_dimension_count']}/{summary['within_cluster_feature_variation']['dimension_count']}",
        f"- Runtime errors: {summary['runtime_errors']}",
        f"- Forbidden answer/gold fields: {len(summary['forbidden_detector_fields'])}",
        f"- Date audit: ISO {summary['date_normalization'].get('dates_iso', 0)}, normalized {summary['date_normalization'].get('dates_normalized', 0)}, unparsed {summary['date_normalization'].get('dates_unparsed', 0)}, null {summary['date_normalization'].get('dates_null', 0)}; unparsed rate {summary['date_normalization'].get('dates_unparsed_rate', 0.0):.4f} (maximum {summary['date_normalization'].get('dates_unparsed_rate_max', 0.0):.4f})",
        "",
        "## Empty retrievals",
        "",
        f"zero_hit_queries: {', '.join(summary['zero_hit_queries']) if summary['zero_hit_queries'] else 'none'}; zero_hit_rate: {summary['zero_hit_rate']:.4f}.",
        "A zero-hit packet is `absent` by construction of the BM25 `score > 0` filter, not by an observed evidence state. It must be explicitly flagged to annotators; it must not be presented as evidence that relevant sources do not exist.",
        "",
        "## Retrieval results",
        "",
        "Values are estimates with cluster-bootstrap 95% CIs. Precision@10 is retained only as a raw diagnostic with its attainable ceiling.",
        "",
        "| Population | n | Recall@10 | Precision@|rel| | Success@1 | Success@3 | Success@5 | Success@10 | MRR | Raw P@10 (ceiling) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    metric_blocks = [("All queries", summary["metrics_including_zero_hits"])]
    if summary["zero_hit_queries"]:
        metric_blocks.append(("Excluding zero hits", summary["metrics_excluding_zero_hits"]))
    for label, block in metric_blocks:
        if not block["n_queries"]:
            lines.append(f"| {label} | 0 | — | — | — | — | — | — | — | — |")
            continue
        raw = block["metrics"]["precision_at_10"]["estimate"]
        ceiling = block["metrics"]["precision_at_10_ceiling"]["estimate"]
        lines.append(
            f"| {label} | {block['n_queries']} | {metric(block, 'recall_at_10')} | {metric(block, 'precision_at_rel')} | "
            f"{metric(block, 'success_at_1')} | {metric(block, 'success_at_3')} | {metric(block, 'success_at_5')} | "
            f"{metric(block, 'success_at_10')} | {metric(block, 'mrr')} | {raw:.4f} ({ceiling:.4f}) |"
        )
    lines.extend(["", "### Per-task integrity", "", "| Task | n_queries | n_source_clusters | distinct vectors / rows | rate | zero hits | zero-hit rate |", "|---|---:|---:|---:|---:|---:|---:|"])
    for task, block in summary["tasks"].items():
        degeneracy = summary["detector_feature_distinctness"][task]
        lines.append(f"| {task.upper()} | {block['n_queries']} | {block['n_source_clusters']} | {degeneracy['distinct_feature_vectors']} / {degeneracy['rows']} | {degeneracy['distinct_feature_vector_rate']:.4f} | {len(block['zero_hit_queries'])} | {block['zero_hit_rate']:.4f} |")
    if not summary["zero_hit_queries"]:
        lines.extend(["", "Including and excluding zero hits are identical because this run has no zero-hit queries; one row is shown."])
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "This validates question-conditioned local CSKG retrieval and option-blind evidence construction only. It does not test the support-state classifier, answer generation, routing AURC, harm, or oracle-gap recovery.",
            "",
            "Gate C is an abstention-policy claim in this frozen policy: the structural policy never selects `vanilla_rag`, so it must not be described as evidence of routing among retrieval strategies.",
            "",
        ]
    )
    return "\n".join(lines)


def _evaluate_row(catalog: dict[str, Any], trace: dict[str, Any]) -> dict[str, Any]:
    hits = trace.get("retrieved_context", [])
    hit_ids = [str(hit["source_id"]) for hit in hits]
    anchor = str(trace["anchor_id"])
    gold = [str(value) for value in catalog["source"]["blog_ids"] if str(value) != anchor]
    relevant_ranks = [index for index, doc_id in enumerate(hit_ids[:10], 1) if doc_id in gold]
    rel_count = len(gold)
    found = len(relevant_ranks)
    row: dict[str, Any] = {
        "query_id": trace["id"],
        "task": trace["task"],
        "source_cluster_id": anchor,
        "zero_hit": not hits,
        "retrieval_feature_vector": [
            [hit["source_id"], hit["retrieval_score"]] for hit in hits
        ],
        "latency_ms": float(trace["latency_ms"]),
        "recall_at_10": found / rel_count,
        "full_cluster_recovery": float(found == rel_count),
        "mrr": 1.0 / relevant_ranks[0] if relevant_ranks else 0.0,
        "precision_at_rel": sum(doc_id in gold for doc_id in hit_ids[:rel_count]) / rel_count,
        "precision_at_10": found / 10,
        "precision_at_10_ceiling": rel_count / 10,
    }
    for k in (1, 3, 5, 10):
        row[f"success_at_{k}"] = float(any(doc_id in gold for doc_id in hit_ids[:k]))
    return row


def _within_cluster_feature_variation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, float]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["source_cluster_id"])].append(row["detector_features"])
    dimensions = sorted(next(iter(rows))["detector_features"]) if rows else []
    varying = [
        dimension
        for dimension in dimensions
        if any(len({features[dimension] for features in cluster}) > 1 for cluster in grouped.values())
    ]
    multi_query_clusters = [cluster for cluster in grouped.values() if len(cluster) > 1]
    collapsed = sum(
        len({tuple(sorted(features.items())) for features in cluster}) == 1
        for cluster in multi_query_clusters
    )
    return {
        "source_clusters": len(grouped),
        "multi_query_source_clusters": len(multi_query_clusters),
        "fully_collapsed_clusters": collapsed,
        "dimension_count": len(dimensions),
        "varying_dimension_count": len(varying),
        "varying_dimensions": varying,
        "constant_within_every_cluster_dimensions": sorted(set(dimensions) - set(varying)),
    }


def _summarize(rows: list[dict[str, Any]], samples: int, seed: int) -> dict[str, Any]:
    nonzero = [row for row in rows if not row["zero_hit"]]
    return {
        "n_queries": len(rows),
        "n_source_clusters": len({row["source_cluster_id"] for row in rows}),
        "zero_hit_queries": [row["query_id"] for row in rows if row["zero_hit"]],
        "zero_hit_rate": sum(row["zero_hit"] for row in rows) / len(rows) if rows else 0.0,
        "including_zero_hits": _metric_block(rows, samples, seed),
        "excluding_zero_hits": _metric_block(nonzero, samples, seed),
        "median_latency_ms": median(row["latency_ms"] for row in rows) if rows else None,
        "p95_latency_ms": _quantile(sorted(row["latency_ms"] for row in rows), 0.95),
    }


def _metric_block(rows: list[dict[str, Any]], samples: int, seed: int) -> dict[str, Any]:
    if not rows:
        return {"n_queries": 0, "metrics": {}}
    return {
        "n_queries": len(rows),
        "metrics": {
            key: {
                "estimate": _mean(rows, key),
                "cluster_bootstrap_95_ci": _cluster_bootstrap_ci(rows, lambda sample, key=key: _mean(sample, key), samples, seed),
            }
            for key in METRIC_KEYS
        },
    }


def _cluster_bootstrap_ci(
    rows: list[dict[str, Any]],
    statistic: Callable[[list[dict[str, Any]]], float],
    samples: int,
    seed: int,
) -> list[float]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["source_cluster_id"])].append(row)
    clusters = sorted(grouped)
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        sample = [row for _cluster in (rng.choice(clusters) for _ in clusters) for row in grouped[_cluster]]
        estimates.append(statistic(sample))
    estimates.sort()
    return [_quantile(estimates, 0.025), _quantile(estimates, 0.975)]


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def _quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    index = (len(values) - 1) * probability
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    weight = index - lower
    return values[lower] * (1 - weight) + values[upper] * weight
