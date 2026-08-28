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
from .features import extract_features
from .io import read_jsonl, write_json, write_jsonl
from .metrics import classification_metrics
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
    for query_id, row_features in zip(test_ids, features(test_ids)):
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
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "predictions.jsonl", predictions)
    write_json(args.output_dir / "summary.json", {"partition": args.partition, "rows": len(predictions), "metrics": metrics})


def import_cticonnect(args: argparse.Namespace) -> None:
    release = CTIConnectRelease.load(args.cticonnect_root, verify_hashes=not args.skip_hash_check)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "catalog.jsonl", release.catalog_rows())
    write_json(args.output_dir / "import_summary.json", release.summary())


def make_detector_input(args: argparse.Namespace) -> None:
    release = CTIConnectRelease.load(args.cticonnect_root, verify_hashes=not args.skip_hash_check)
    catalog = release.catalog_rows()
    traces = read_jsonl(args.retrieval_traces)
    rows, summary = build_detector_inputs(catalog, traces, release.report_index())
    write_jsonl(args.output, rows)
    write_json(args.summary, summary)


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
        retriever = ShippedCSKGRetriever(args.cticonnect_root, anchors, release.report_index())
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
            trace["transform"] = {"kind": "shipped_cskg_anchor_entity_vocab", "llm_calls": 0}
        traces.append(trace)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "retrieval.jsonl", traces)
    manifest = {
        "experiment_id": "PX-034",
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
    detector_input.set_defaults(func=make_detector_input)
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


if __name__ == "__main__":
    main()
