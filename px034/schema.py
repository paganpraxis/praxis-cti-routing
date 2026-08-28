from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class SupportState(str, Enum):
    DECISIVE = "decisive"
    COMPLEMENTARY = "complementary"
    CONFLICTING = "conflicting"
    STALE = "stale"
    ABSENT = "absent"
    OTHER = "other"


class Route(str, Enum):
    CLOSED_BOOK = "closed_book"
    VANILLA_RAG = "vanilla_rag"
    DOMAIN_SPECIFIC = "domain_specific"
    ABSTAIN = "abstain"


FORBIDDEN_DETECTOR_FIELDS = frozenset(
    {
        "answer",
        "answers",
        "answer_options",
        "candidate_answer",
        "candidate_answers",
        "correct_answer",
        "expected_output",
        "gold",
        "gold_answer",
        "gold_label",
        "label",
        "support_state",
    }
)


@dataclass(frozen=True)
class Evidence:
    source_id: str
    text: str
    title: str = ""
    source_type: str = "unknown"
    published_at: str | None = None
    retrieved_at: str | None = None
    retrieval_score: float | None = None
    canonical_entities: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Evidence":
        return cls(
            source_id=str(value["source_id"]),
            text=str(value.get("text", "")),
            title=str(value.get("title", "")),
            source_type=str(value.get("source_type", "unknown")),
            published_at=value.get("published_at"),
            retrieved_at=value.get("retrieved_at"),
            retrieval_score=_optional_float(value.get("retrieval_score")),
            canonical_entities=tuple(str(x) for x in value.get("canonical_entities", [])),
        )


@dataclass(frozen=True)
class DetectorInput:
    query_id: str
    question: str
    task: str
    evidence: tuple[Evidence, ...]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DetectorInput":
        assert_option_blind(value)
        return cls(
            query_id=str(value["query_id"]),
            question=str(value["question"]),
            task=str(value["task"]),
            evidence=tuple(Evidence.from_dict(x) for x in value.get("evidence", [])),
        )


def assert_option_blind(value: Any, path: str = "$") -> None:
    """Reject answer- or label-bearing fields anywhere in detector input."""
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).lower().strip()
            if normalized in FORBIDDEN_DETECTOR_FIELDS:
                raise ValueError(f"option-blind detector input contains forbidden field {path}.{key}")
            assert_option_blind(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            assert_option_blind(child, f"{path}[{index}]")


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)
