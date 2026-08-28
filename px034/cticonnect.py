from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .io import read_jsonl
from .schema import DetectorInput


TASK_CATEGORIES = {
    "rcm": "entity_linking",
    "wim": "entity_linking",
    "atd": "entity_linking",
    "esd": "entity_linking",
    "ata": "entity_attribution",
    "vca": "entity_attribution",
    "csc": "multi_doc_synthesis",
    "tap": "multi_doc_synthesis",
    "mla": "multi_doc_synthesis",
}
CATEGORY_CANONICAL = {"multi_doc_synthesis": "multi_document_synthesis"}
PAPER_RELEASE_COUNT = 1860
CTI_ENTITY = re.compile(
    r"\b(?:CVE-\d{4}-\d+|CWE-\d+|CAPEC-\d+|T\d{4}(?:\.\d{3})?|G\d{4}|S\d{4}|BLOG-\d+)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CTIConnectRelease:
    root: Path
    manifest: dict[str, Any]
    records: tuple[dict[str, Any], ...]

    @classmethod
    def load(cls, root: Path, verify_hashes: bool = True) -> "CTIConnectRelease":
        manifest_path = root / "data" / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"CTIConnect manifest not found: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        records: list[dict[str, Any]] = []
        seen: set[str] = set()
        for task, spec in manifest["tasks"].items():
            if task not in TASK_CATEGORIES:
                raise ValueError(f"unsupported CTIConnect task in manifest: {task}")
            path = root / spec["relative_path"]
            if verify_hashes and _sha256_file(path) != spec["sha256"]:
                raise ValueError(f"SHA-256 mismatch for {path}")
            task_rows = read_jsonl(path)
            if len(task_rows) != int(spec["count"]):
                raise ValueError(f"count mismatch for {task}: manifest={spec['count']} actual={len(task_rows)}")
            for row in task_rows:
                _validate_qa(row, task, spec)
                if row["id"] in seen:
                    raise ValueError(f"duplicate CTIConnect query id: {row['id']}")
                seen.add(row["id"])
                records.append(row)
        if len(records) != int(manifest["total_count"]):
            raise ValueError(f"release total mismatch: manifest={manifest['total_count']} actual={len(records)}")
        return cls(root=root, manifest=manifest, records=tuple(records))

    def summary(self) -> dict[str, Any]:
        task_counts = Counter(str(row["task"]) for row in self.records)
        category_counts = Counter(str(row["category"]) for row in self.records)
        return {
            "name": self.manifest.get("name"),
            "version": self.manifest.get("version"),
            "released": self.manifest.get("released"),
            "manifest_count": self.manifest["total_count"],
            "actual_count": len(self.records),
            "paper_release_count": PAPER_RELEASE_COUNT,
            "paper_release_count_delta": len(self.records) - PAPER_RELEASE_COUNT,
            "task_counts": dict(sorted(task_counts.items())),
            "category_counts": dict(sorted(category_counts.items())),
            "manifest_sha256": _sha256_file(self.root / "data" / "manifest.json"),
        }

    def catalog_rows(self) -> list[dict[str, Any]]:
        cluster_ids = _source_cluster_ids(self.records)
        result = []
        for row in self.records:
            result.append(
                {
                    "query_id": row["id"],
                    "task": row["task"],
                    "category": row["category"],
                    "eval_type": row["eval_type"],
                    "question": row["question"],
                    "answer": row["answer"],
                    "ground_truth": row["ground_truth"],
                    "source": row["source"],
                    "source_cluster_id": cluster_ids[row["id"]],
                }
            )
        return result

    def report_index(self) -> dict[str, dict[str, Any]]:
        path = self.root / "corpus_reports" / "preprocessed_reports.jsonl"
        index = {}
        for row in read_jsonl(path):
            source_id = f"BLOG-{row['id']}"
            index[source_id] = {
                "source_id": source_id,
                "title": str(row.get("title", "")),
                "source_type": "vendor_report",
                "text": str(row.get("preprocessed", "")),
                "published_at": row.get("publish_date"),
                "metadata": row.get("metadata", {}),
            }
        return index


def build_detector_inputs(
    catalog_rows: Iterable[dict[str, Any]],
    retrieval_rows: Iterable[dict[str, Any]],
    report_index: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Join retrieval traces to public questions without exposing catalog gold."""
    public = {
        str(row["query_id"]): {
            "query_id": str(row["query_id"]),
            "question": str(row["question"]),
            "task": CATEGORY_CANONICAL.get(str(row["category"]), str(row["category"])),
        }
        for row in catalog_rows
    }
    output = []
    seen: set[str] = set()
    missing_queries = []
    for trace in retrieval_rows:
        query_id = str(trace.get("query_id", trace.get("id", "")))
        if query_id not in public:
            missing_queries.append(query_id)
            continue
        if query_id in seen:
            raise ValueError(f"duplicate retrieval trace for {query_id}")
        seen.add(query_id)
        contexts = trace.get("retrieved_context", trace.get("evidence"))
        if contexts is None:
            raise ValueError(f"retrieval trace {query_id} has no retrieved_context/evidence")
        evidence = [
            _normalize_context({**item, "retrieved_at": item.get("retrieved_at", trace.get("retrieved_at"))}, report_index or {})
            for item in contexts
        ]
        row = {**public[query_id], "evidence": evidence}
        DetectorInput.from_dict(row)
        output.append(row)
    missing = sorted(set(public) - seen)
    unknown = sorted(set(missing_queries))
    return output, {
        "catalog_queries": len(public),
        "retrieval_traces": len(seen),
        "detector_inputs": len(output),
        "missing_retrieval_query_count": len(missing),
        "missing_retrieval_queries_sample": missing[:100],
        "missing_retrieval_queries_truncated": len(missing) > 100,
        "unknown_retrieval_query_count": len(unknown),
        "unknown_retrieval_queries": unknown[:100],
        "unknown_retrieval_queries_truncated": len(unknown) > 100,
    }


def _normalize_context(value: dict[str, Any], report_index: dict[str, dict[str, Any]]) -> dict[str, Any]:
    source_id = str(value.get("source_id", value.get("doc_id", value.get("id", ""))))
    base = report_index.get(source_id, {})
    text = value.get("text", value.get("contents", value.get("preprocessed", base.get("text", ""))))
    result = {
        "source_id": source_id,
        "title": str(value.get("title", base.get("title", ""))),
        "source_type": str(value.get("source_type", base.get("source_type", "unknown"))),
        "text": str(text),
        "published_at": value.get("published_at", value.get("publish_date", base.get("published_at"))),
        "retrieved_at": value.get("retrieved_at"),
        "retrieval_score": value.get("retrieval_score", value.get("score")),
        "canonical_entities": sorted(set(str(x).upper() for x in value.get("canonical_entities", CTI_ENTITY.findall(str(text))))),
    }
    return result


def _validate_qa(row: dict[str, Any], task: str, spec: dict[str, Any]) -> None:
    required = {"id", "task", "category", "eval_type", "question", "answer", "ground_truth", "source"}
    missing = required - row.keys()
    if missing:
        raise ValueError(f"{row.get('id', task)} missing fields: {sorted(missing)}")
    if row["task"] != task or row["category"] != TASK_CATEGORIES[task]:
        raise ValueError(f"task/category mismatch for {row['id']}")
    if row["eval_type"] != spec["eval_type"]:
        raise ValueError(f"eval_type mismatch for {row['id']}")


def _source_cluster_ids(records: Iterable[dict[str, Any]]) -> dict[str, str]:
    """Connect records sharing any source document to prevent split leakage."""
    parent: dict[str, str] = {}

    def find(value: str) -> str:
        parent.setdefault(value, value)
        if parent[value] != value:
            parent[value] = find(parent[value])
        return parent[value]

    def union(left: str, right: str) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[max(a, b)] = min(a, b)

    record_sources: dict[str, list[str]] = {}
    for row in records:
        source = row["source"]
        ids = [str(x) for x in (source.get("blog_ids") or [])]
        if source.get("source_id"):
            ids.append(str(source["source_id"]))
        if not ids:
            ids = [f"query:{row['id']}"]
        tokens = [f"source:{value}" for value in sorted(set(ids))]
        for token in tokens[1:]:
            union(tokens[0], token)
        record_sources[str(row["id"])] = tokens
    return {query_id: find(tokens[0]) for query_id, tokens in record_sources.items()}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
