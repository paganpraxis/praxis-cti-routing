from __future__ import annotations

import math
import re
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Protocol


TOKEN = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class Hit:
    source_id: str
    title: str
    text: str
    score: float
    source_type: str = "unknown"
    published_at: str | None = None
    matched_tokens: tuple[str, ...] = ()
    canonical_entities: tuple[str, ...] = ()

    def to_dict(self, rank: int) -> dict[str, Any]:
        return {
            "rank": rank,
            "source_id": self.source_id,
            "title": self.title,
            "text": self.text,
            "retrieval_score": self.score,
            "source_type": self.source_type,
            "published_at": self.published_at,
            "matched_tokens": list(self.matched_tokens),
            "canonical_entities": list(self.canonical_entities),
        }


class Retriever(Protocol):
    def retrieve(self, query: str, task: str, k: int) -> list[Hit]: ...


Transform = Callable[[str, str], tuple[list[str], dict[str, Any]]]


def execute_trace(
    record: dict[str, Any],
    strategy: str,
    retriever: Retriever,
    k: int,
    transform: Transform | None = None,
) -> dict[str, Any]:
    """Execute retrieval only; candidate answer generation is out of scope."""
    started = time.perf_counter()
    retrieved_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    question = str(record["question"])
    task = str(record["task"])
    queries = [question]
    transform_meta: dict[str, Any] = {"kind": "identity"}
    if transform is not None:
        transform_started = time.perf_counter()
        queries, transform_meta = transform(question, task)
        transform_meta = dict(transform_meta)
        transform_meta["latency_ms"] = (time.perf_counter() - transform_started) * 1000
    if not queries:
        queries = [question]
        transform_meta["fallback_to_question"] = True

    events = []
    best_hits: dict[str, Hit] = {}
    for index, query in enumerate(queries):
        event_started = time.perf_counter()
        hits = retriever.retrieve(query, task, k)
        events.append(
            {
                "query_index": index,
                "query": query,
                "latency_ms": (time.perf_counter() - event_started) * 1000,
                "hits": [hit.to_dict(rank) for rank, hit in enumerate(hits, 1)],
            }
        )
        for hit in hits:
            current = best_hits.get(hit.source_id)
            if current is None or hit.score > current.score:
                best_hits[hit.source_id] = hit
    merged = sorted(best_hits.values(), key=lambda hit: (-hit.score, hit.source_id))
    return {
        "id": str(record["query_id"]),
        "task": task,
        "category": str(record["category"]),
        "retrieval_strategy": strategy,
        "retrieved_at": retrieved_at,
        "top_k_per_query": k,
        "transform": transform_meta,
        "retrieval_events": events,
        "retrieved_context": [hit.to_dict(rank) for rank, hit in enumerate(merged, 1)],
        "latency_ms": (time.perf_counter() - started) * 1000,
    }


class EntityBM25:
    """Small, dependency-free BM25 implementation for the shipped CSKG vocab."""

    def __init__(self, documents: dict[str, Iterable[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_ids = sorted(documents)
        self.docs = {doc_id: _tokens(documents[doc_id]) for doc_id in self.doc_ids}
        self.avgdl = sum(map(len, self.docs.values())) / len(self.docs) if self.docs else 0.0
        document_frequency = Counter(token for tokens in self.docs.values() for token in set(tokens))
        count = len(self.docs)
        self.idf = {token: math.log(count - freq + 0.5) - math.log(freq + 0.5) for token, freq in document_frequency.items()}
        average_idf = sum(self.idf.values()) / len(self.idf) if self.idf else 0.0
        floor = 0.25 * average_idf
        self.idf = {token: (floor if value < 0 else value) for token, value in self.idf.items()}

    def search(self, surfaces: Iterable[str], k: int, exclude: Iterable[str] = ()) -> list[tuple[str, float, tuple[str, ...]]]:
        query = _tokens(surfaces)
        excluded = set(exclude)
        scores = []
        for doc_id in self.doc_ids:
            if doc_id in excluded:
                continue
            tokens = self.docs[doc_id]
            frequencies = Counter(tokens)
            score = 0.0
            for token in query:
                frequency = frequencies[token]
                if not frequency:
                    continue
                denominator = frequency + self.k1 * (1 - self.b + self.b * len(tokens) / (self.avgdl or 1.0))
                score += self.idf.get(token, 0.0) * frequency * (self.k1 + 1) / denominator
            if score > 0:
                scores.append((doc_id, score, tuple(sorted(set(query) & set(tokens)))))
        return sorted(scores, key=lambda item: (-item[1], item[0]))[:k]


def _tokens(surfaces: Iterable[str]) -> list[str]:
    return [token for surface in surfaces for token in TOKEN.findall(str(surface).lower())]
