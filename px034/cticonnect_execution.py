from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from .cticonnect import CTIConnectRelease
from .io import read_jsonl
from .retrieval import EntityBM25, Hit


class OfficialKBRetriever:
    """Adapter over the official CTIConnect dense KBRetriever."""

    def __init__(self, repository: Path, cache_dir: Path | None = None):
        _add_repository(repository)
        from baselines._shared.retriever import KBRetriever

        self.inner = KBRetriever(cache_dir=cache_dir)

    def retrieve(self, query: str, task: str, k: int) -> list[Hit]:
        return [
            Hit(source_id=str(item.doc_id), title=str(item.title), text=str(item.text), score=float(item.score), source_type=_target_type(task))
            for item in self.inner.retrieve_for_task(query, task, k=k)
        ]


class OfficialTransformer:
    """Use the release's exact EtR/DtR prompts with its ChatLLM wrapper."""

    def __init__(self, repository: Path, strategy: str, model: str | None = None):
        _add_repository(repository)
        from baselines._shared.llm import ChatLLM

        self.strategy = strategy
        self.llm = ChatLLM(model=model) if model else ChatLLM()

    def __call__(self, question: str, task: str) -> tuple[list[str], dict[str, Any]]:
        if self.strategy == "etr":
            from baselines.etr.run import build_extract_prompt

            value = self.llm.chat(build_extract_prompt(question, task), max_tokens=128).strip()
            return ([value] if value else []), {"kind": "extract_then_retrieve", "model": self.llm.model, "raw_output": value}
        if self.strategy == "dtr":
            from baselines.dtr.run import DECOMPOSE_PROMPT

            value = self.llm.chat(DECOMPOSE_PROMPT.format(question=question), max_tokens=256)
            behaviors = [line.strip("-* \t") for line in value.splitlines() if line.strip()][:5]
            return behaviors, {"kind": "decompose_then_retrieve", "model": self.llm.model, "raw_output": value}
        raise ValueError(f"unsupported official transform: {self.strategy}")


class ShippedCSKGRetriever:
    """Question-conditioned search over the shipped CSKG entity vocabulary."""

    DEFAULT_QUESTION_WEIGHT = 0.10

    def __init__(
        self,
        repository: Path,
        anchor_by_query: dict[str, str],
        report_index: dict[str, dict[str, Any]],
        question_weight: float = DEFAULT_QUESTION_WEIGHT,
    ):
        if not 0.0 <= question_weight <= 1.0:
            raise ValueError("question_weight must be between 0 and 1")
        vocab_rows = read_jsonl(repository / "cskg" / "per_doc_entities.jsonl")
        self.surfaces = {str(row["doc_id"]): list(row["entity_surfaces"]) for row in vocab_rows}
        self.index = EntityBM25(self.surfaces)
        self.anchor_by_query = anchor_by_query
        self.report_index = report_index
        self.question_weight = question_weight
        self.current_query_id: str | None = None

    def set_query(self, query_id: str) -> None:
        self.current_query_id = query_id

    def retrieve(self, query: str, task: str, k: int) -> list[Hit]:
        if not self.current_query_id or self.current_query_id not in self.anchor_by_query:
            raise ValueError(f"no MDS anchor for query {self.current_query_id}")
        anchor = self.anchor_by_query[self.current_query_id]
        if anchor not in self.surfaces:
            raise ValueError(f"anchor {anchor} is absent from shipped CSKG")
        excluded = {anchor}
        candidate_count = max(0, len(self.surfaces) - len(excluded))
        anchor_hits = self.index.search(self.surfaces[anchor], k=candidate_count, exclude=excluded)
        question_hits = self.index.search([query], k=candidate_count, exclude=excluded)
        anchor_scores = {doc_id: score for doc_id, score, _ in anchor_hits}
        question_scores = {doc_id: score for doc_id, score, _ in question_hits}
        anchor_max = max(anchor_scores.values(), default=1.0)
        question_max = max(question_scores.values(), default=1.0)
        matched_by_doc: dict[str, set[str]] = {}
        for doc_id, _, matched in anchor_hits + question_hits:
            matched_by_doc.setdefault(doc_id, set()).update(matched)
        fused = []
        for doc_id in anchor_scores.keys() | question_scores.keys():
            score = (
                (1.0 - self.question_weight) * anchor_scores.get(doc_id, 0.0) / anchor_max
                + self.question_weight * question_scores.get(doc_id, 0.0) / question_max
            )
            if score > 0:
                fused.append((doc_id, score, tuple(sorted(matched_by_doc[doc_id]))))

        result = []
        for doc_id, score, matched in sorted(fused, key=lambda item: (-item[1], item[0]))[:k]:
            report = self.report_index.get(doc_id, {})
            result.append(
                Hit(
                    source_id=doc_id,
                    title=str(report.get("title", "")),
                    text=str(report.get("text", "")),
                    score=score,
                    source_type="vendor_report",
                    published_at=report.get("published_at"),
                    matched_tokens=matched,
                    canonical_entities=tuple(self.surfaces.get(doc_id, [])),
                )
            )
        return result


def load_anchor_manifest(path: Path) -> dict[str, str]:
    result = {}
    for row in read_jsonl(path):
        query_id, anchor_id = str(row["query_id"]), str(row["anchor_id"])
        if query_id in result:
            raise ValueError(f"duplicate anchor mapping for {query_id}")
        result[query_id] = anchor_id
    return result


def exploratory_first_source_anchors(release: CTIConnectRelease) -> dict[str, str]:
    result = {}
    for row in release.records:
        blog_ids = row["source"].get("blog_ids") or []
        if blog_ids:
            result[str(row["id"])] = str(blog_ids[0])
    return result


def _add_repository(repository: Path) -> None:
    value = str(repository.resolve())
    if value not in sys.path:
        sys.path.insert(0, value)


def _target_type(task: str) -> str:
    return {"rcm": "cwe", "wim": "cve", "atd": "mitre", "esd": "capec", "ata": "mitre", "vca": "cwe"}[task]
