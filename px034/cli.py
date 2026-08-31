from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from .cticonnect import CTIConnectRelease, build_detector_inputs
from .cticonnect_execution import (
    OfficialKBRetriever,
    OfficialTransformer,
    ShippedCSKGRetriever,
    exploratory_first_source_anchors,
    load_anchor_manifest,
)
from .detector import CentroidDetector
from .evaluation import evaluate_mds_retrieval, render_mds_report
from .features import extract_features
from .io import read_jsonl, write_json, write_jsonl
from .metrics import classification_metrics, feature_degeneracy
from .routing import route_for_state
from .retrieval import execute_trace
from .schema import DetectorInput, SupportState
from .split import freeze_hash, grouped_stratified_split, stratified_split


def _labels(path: Path) -> dict[str, str]:
    labels = {}
    for row in read_jsonl(path):
        state = SupportState(str(row["support_state"]))
        if state == SupportState.OTHER:
            continue
        labels[str(row["query_id"])] = state.value
    return labels


def make_splits(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.annotations)
    grouped = bool(rows and all(row.get("source_cluster_id") for row in rows))
    split = grouped_stratified_split(rows, args.seed) if grouped else stratified_split(rows, args.seed)
    payload = {"seed": args.seed, "source_grouped": grouped, "splits": split}
    payload["sha256"] = freeze_hash(payload)
    write_json(args.output, payload)


def run_baseline(args: argparse.Namespace) -> None:
    detector_rows = {str(row["query_id"]): DetectorInput.from_dict(row) for row in read_jsonl(args.detector_input)}
    labels = _labels(args.annotations)
    split_payload = json.loads(args.splits.read_text(encoding="utf-8"))
    as_of = datetime.fromisoformat(args.as_of)

    def features(ids: list[str]) -> list[dict[str, float]]:
        return [extract_features(detector_rows[query_id], as_of) for query_id in ids]

    train_ids = [x for x in split_payload["splits"]["train"] if x in labels]
    test_ids = [x for x in split_payload["splits"][args.partition] if x in labels]
    model = CentroidDetector.fit(features(train_ids), [labels[x] for x in train_ids])
    predictions = []
    test_features = features(test_ids)
    for query_id, row_features in zip(test_ids, test_features):
        prediction, confidence = model.predict_one(row_features)
        routing = route_for_state(prediction, detector_rows[query_id].task)
        predictions.append(
            {
                "query_id": query_id,
                "gold_support_state": labels[query_id],
                "predicted_support_state": prediction,
                "confidence": confidence,
                **routing,
                "features": row_features,
            }
        )
    metrics = classification_metrics(
        [row["gold_support_state"] for row in predictions],
        [row["predicted_support_state"] for row in predictions],
    )
    degeneracy_rows = [
        {"task": detector_rows[query_id].task, "features": row_features}
        for query_id, row_features in zip(test_ids, test_features)
    ]
    degeneracy = feature_degeneracy(degeneracy_rows)
    _enforce_feature_floor(degeneracy, args.feature_distinct_floor)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "predictions.jsonl", predictions)
    write_json(
        args.output_dir / "summary.json",
        {
            "partition": args.partition,
            "rows": len(predictions),
            "metrics": metrics,
            "feature_degeneracy": degeneracy,
            "feature_distinct_floor": args.feature_distinct_floor,
        },
    )


def import_cticonnect(args: argparse.Namespace) -> None:
    release = CTIConnectRelease.load(args.cticonnect_root, verify_hashes=not args.skip_hash_check)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "catalog.jsonl", release.catalog_rows())
    write_json(args.output_dir / "import_summary.json", release.summary())


def make_detector_input(args: argparse.Namespace) -> None:
    release = CTIConnectRelease.load(args.cticonnect_root, verify_hashes=not args.skip_hash_check)
    catalog = release.catalog_rows()
    traces = read_jsonl(args.retrieval_traces)
    rows, summary = build_detector_inputs(
        catalog,
        traces,
        release.report_index(),
        dates_unparsed_rate_max=args.dates_unparsed_rate_max,
    )
    write_jsonl(args.output, rows)
    write_json(args.summary, summary)


def evaluate_mds(args: argparse.Namespace) -> None:
    date_normalization = json.loads(args.detector_import_summary.read_text(encoding="utf-8"))
    summary = evaluate_mds_retrieval(
        read_jsonl(args.catalog),
        read_jsonl(args.retrieval_traces),
        read_jsonl(args.detector_input),
        as_of=datetime.fromisoformat(args.as_of),
        date_normalization={key: value for key, value in date_normalization.items() if key.startswith("date")},
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        feature_distinct_floor=args.feature_distinct_floor,
    )
    manifest = json.loads(args.run_manifest.read_text(encoding="utf-8"))
    write_json(args.summary, summary)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_mds_report(summary, manifest), encoding="utf-8")


def run_retrieval(args: argparse.Namespace) -> None:
    release = CTIConnectRelease.load(args.cticonnect_root, verify_hashes=not args.skip_hash_check)
    catalog = release.catalog_rows()
    allowed_tasks = set(args.tasks or [])
    if allowed_tasks:
        unknown = allowed_tasks - set(release.manifest["tasks"])
        if unknown:
            raise ValueError(f"unknown CTIConnect tasks: {sorted(unknown)}")
        catalog = [row for row in catalog if row["task"] in allowed_tasks]
    if args.limit is not None:
        catalog = catalog[: args.limit]

    transformer = None
    if args.strategy == "cskg":
        if args.anchor_manifest:
            anchors = load_anchor_manifest(args.anchor_manifest)
            anchor_policy = "explicit_manifest"
        elif args.exploratory_first_source_anchor:
            anchors = exploratory_first_source_anchors(release)
            anchor_policy = "exploratory_first_gold_cluster_source"
        else:
            raise ValueError("CSKG requires --anchor-manifest; use --exploratory-first-source-anchor only for non-confirmatory work")
        retriever = ShippedCSKGRetriever(
            args.cticonnect_root,
            anchors,
            release.report_index(),
            question_weight=args.cskg_question_weight,
        )
        expected = {"csc", "tap", "mla"}
    else:
        retriever = OfficialKBRetriever(args.cticonnect_root, args.cache_dir)
        anchors = {}
        anchor_policy = None
        expected = {"rcm", "wim", "atd", "esd", "ata", "vca"}
        if args.strategy in {"etr", "dtr"}:
            transformer = OfficialTransformer(args.cticonnect_root, args.strategy, args.transform_model)
    wrong = sorted({row["task"] for row in catalog} - expected)
    if wrong:
        raise ValueError(f"strategy {args.strategy} does not support tasks {wrong}")

    started = time.time()
    traces = []
    for row in catalog:
        if args.strategy == "cskg":
            retriever.set_query(str(row["query_id"]))
        trace = execute_trace(row, args.strategy, retriever, args.top_k, transformer)
        if args.strategy == "cskg":
            trace["anchor_id"] = anchors[str(row["query_id"])]
            trace["anchor_policy"] = anchor_policy
            trace["transform"] = {
                "kind": "question_anchor_fusion_over_shipped_cskg_entity_vocab",
                "question_weight": args.cskg_question_weight,
                "anchor_weight": 1.0 - args.cskg_question_weight,
                "llm_calls": 0,
            }
        traces.append(trace)
    degeneracy_rows = [
        {
            "task": trace["task"],
            "features": [
                [hit["source_id"], hit["retrieval_score"]]
                for hit in trace["retrieved_context"]
            ],
        }
        for trace in traces
    ]
    retrieval_distinctness = feature_degeneracy(degeneracy_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "retrieval.jsonl", traces)
    manifest = {
        "experiment_id": "PX-034",
        "result_version": args.result_version,
        "strategy": args.strategy,
        "tasks": sorted({row["task"] for row in catalog}),
        "top_k": args.top_k,
        "rows": len(traces),
        "started_unix": started,
        "elapsed_seconds": time.time() - started,
        "cticonnect": release.summary(),
        "cticonnect_git_commit": _git_head(args.cticonnect_root),
        "transform_model": args.transform_model,
        "anchor_policy": anchor_policy,
        "cskg_fusion": (
            {
                "question_weight": args.cskg_question_weight,
                "anchor_weight": 1.0 - args.cskg_question_weight,
            }
            if args.strategy == "cskg"
            else None
        ),
        "retrieval_vector_distinctness": retrieval_distinctness,
        "detector_feature_distinct_floor": args.feature_distinct_floor,
        "python": platform.python_version(),
    }
    write_json(args.output_dir / "run_manifest.json", manifest)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PX-034 conflict-aware CTI routing")
    subparsers = parser.add_subparsers(required=True)
    split_parser = subparsers.add_parser("make-splits")
    split_parser.add_argument("--annotations", type=Path, required=True)
    split_parser.add_argument("--output", type=Path, required=True)
    split_parser.add_argument("--seed", type=int, default=34057)
    split_parser.set_defaults(func=make_splits)
    importer = subparsers.add_parser("import-cticonnect")
    importer.add_argument("--cticonnect-root", type=Path, required=True)
    importer.add_argument("--output-dir", type=Path, required=True)
    importer.add_argument("--skip-hash-check", action="store_true")
    importer.set_defaults(func=import_cticonnect)
    detector_input = subparsers.add_parser("make-detector-input")
    detector_input.add_argument("--cticonnect-root", type=Path, required=True)
    detector_input.add_argument("--retrieval-traces", type=Path, required=True)
    detector_input.add_argument("--output", type=Path, required=True)
    detector_input.add_argument("--summary", type=Path, required=True)
    detector_input.add_argument("--skip-hash-check", action="store_true")
    detector_input.add_argument("--dates-unparsed-rate-max", type=float, default=0.0)
    detector_input.set_defaults(func=make_detector_input)
    evaluator = subparsers.add_parser("evaluate-mds")
    evaluator.add_argument("--catalog", type=Path, required=True)
    evaluator.add_argument("--retrieval-traces", type=Path, required=True)
    evaluator.add_argument("--detector-input", type=Path, required=True)
    evaluator.add_argument("--detector-import-summary", type=Path, required=True)
    evaluator.add_argument("--as-of", required=True)
    evaluator.add_argument("--run-manifest", type=Path, required=True)
    evaluator.add_argument("--summary", type=Path, required=True)
    evaluator.add_argument("--report", type=Path, required=True)
    evaluator.add_argument("--bootstrap-samples", type=int, default=2000)
    evaluator.add_argument("--seed", type=int, default=34057)
    evaluator.add_argument("--feature-distinct-floor", type=float, default=0.90)
    evaluator.set_defaults(func=evaluate_mds)
    retrieval = subparsers.add_parser("run-retrieval")
    retrieval.add_argument("--cticonnect-root", type=Path, required=True)
    retrieval.add_argument("--strategy", choices=("vanilla", "etr", "dtr", "cskg"), required=True)
    retrieval.add_argument("--tasks", nargs="+")
    retrieval.add_argument("--top-k", type=int, default=5)
    retrieval.add_argument("--limit", type=int)
    retrieval.add_argument("--cache-dir", type=Path)
    retrieval.add_argument("--transform-model")
    retrieval.add_argument("--anchor-manifest", type=Path)
    retrieval.add_argument("--exploratory-first-source-anchor", action="store_true")
    retrieval.add_argument("--cskg-question-weight", type=float, default=ShippedCSKGRetriever.DEFAULT_QUESTION_WEIGHT)
    retrieval.add_argument("--feature-distinct-floor", type=float, default=0.90)
    retrieval.add_argument("--result-version", default="unversioned")
    retrieval.add_argument("--output-dir", type=Path, required=True)
    retrieval.add_argument("--skip-hash-check", action="store_true")
    retrieval.set_defaults(func=run_retrieval)
    baseline = subparsers.add_parser("run-baseline")
    baseline.add_argument("--detector-input", type=Path, required=True)
    baseline.add_argument("--annotations", type=Path, required=True)
    baseline.add_argument("--splits", type=Path, required=True)
    baseline.add_argument("--partition", choices=("development", "heldout"), default="development")
    baseline.add_argument("--as-of", required=True, help="Frozen ISO-8601 feature reference time")
    baseline.add_argument("--output-dir", type=Path, required=True)
    baseline.add_argument("--feature-distinct-floor", type=float, default=0.90)
    baseline.set_defaults(func=run_baseline)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


def _git_head(root: Path) -> str | None:
    head = root / ".git" / "HEAD"
    if not head.exists():
        return None
    value = head.read_text(encoding="utf-8").strip()
    if value.startswith("ref: "):
        ref = root / ".git" / value[5:]
        return ref.read_text(encoding="utf-8").strip() if ref.exists() else None
    return value


def _enforce_feature_floor(summary: dict[str, dict[str, float | int]], floor: float) -> None:
    if not 0.0 <= floor <= 1.0:
        raise ValueError("feature distinct floor must be between 0 and 1")
    failed = {
        task: values["distinct_feature_vector_rate"]
        for task, values in summary.items()
        if task != "overall" and float(values["distinct_feature_vector_rate"]) < floor
    }
    if failed:
        raise ValueError(f"feature degeneracy below declared floor {floor}: {failed}")


if __name__ == "__main__":
    main()
