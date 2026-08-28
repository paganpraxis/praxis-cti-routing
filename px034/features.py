from __future__ import annotations

import math
import re
from collections import Counter
from datetime import datetime, timezone

from .schema import DetectorInput


TOKEN = re.compile(r"[a-z0-9_.:-]+", re.IGNORECASE)
CTI_ID = re.compile(r"\b(?:CVE-\d{4}-\d+|CWE-\d+|CAPEC-\d+|T\d{4}(?:\.\d{3})?|G\d{4}|S\d{4})\b", re.IGNORECASE)
NEGATION = re.compile(r"\b(?:not|never|no longer|incorrect|false|unrelated|disputed)\b", re.IGNORECASE)


def extract_features(row: DetectorInput, as_of: datetime | None = None) -> dict[str, float]:
    """Extract answer-independent evidence-structure features."""
    as_of = as_of or datetime.now(timezone.utc)
    evidence = row.evidence
    scores = [item.retrieval_score for item in evidence if item.retrieval_score is not None]
    entity_sets = [set(map(str.lower, item.canonical_entities)) for item in evidence]
    text_id_sets = [set(x.upper() for x in CTI_ID.findall(item.text)) for item in evidence]
    source_counts = Counter(item.source_type for item in evidence)
    ages = [_age_days(item.published_at, as_of) for item in evidence]
    valid_ages = [age for age in ages if age is not None]
    query_tokens = set(TOKEN.findall(row.question.lower()))
    overlaps = [_jaccard(query_tokens, set(TOKEN.findall(item.text.lower()))) for item in evidence]

    return {
        "evidence_count": float(len(evidence)),
        "source_type_count": float(len(source_counts)),
        "source_concentration": _concentration(source_counts.values()),
        "score_max": max(scores, default=0.0),
        "score_mean": _mean(scores),
        "score_std": _std(scores),
        "score_top2_margin": _top2_margin(scores),
        "query_evidence_overlap_mean": _mean(overlaps),
        "query_evidence_overlap_max": max(overlaps, default=0.0),
        "entity_pair_agreement": _pairwise_jaccard(entity_sets),
        "cti_id_pair_agreement": _pairwise_jaccard(text_id_sets),
        "unique_entity_count": float(len(set().union(*entity_sets)) if entity_sets else 0),
        "unique_cti_id_count": float(len(set().union(*text_id_sets)) if text_id_sets else 0),
        "dated_fraction": len(valid_ages) / len(evidence) if evidence else 0.0,
        "age_days_max": max(valid_ages, default=0.0),
        "age_days_span": (max(valid_ages) - min(valid_ages)) if valid_ages else 0.0,
        "negation_source_fraction": sum(bool(NEGATION.search(x.text)) for x in evidence) / len(evidence) if evidence else 0.0,
    }


def _age_days(value: str | None, as_of: datetime) -> float | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return max(0.0, (as_of - parsed).total_seconds() / 86400)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = _mean(values)
    return math.sqrt(sum((x - avg) ** 2 for x in values) / len(values))


def _top2_margin(values: list[float]) -> float:
    ordered = sorted(values, reverse=True)
    return ordered[0] - ordered[1] if len(ordered) > 1 else (ordered[0] if ordered else 0.0)


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _pairwise_jaccard(values: list[set[str]]) -> float:
    pairs = [_jaccard(values[i], values[j]) for i in range(len(values)) for j in range(i + 1, len(values))]
    return _mean(pairs)


def _concentration(counts: object) -> float:
    values = list(counts)  # type: ignore[arg-type]
    total = sum(values)
    return sum((value / total) ** 2 for value in values) if total else 0.0
