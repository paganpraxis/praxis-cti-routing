from __future__ import annotations

from collections import Counter
from typing import Iterable


def classification_metrics(gold: list[str], predicted: list[str]) -> dict[str, object]:
    if len(gold) != len(predicted) or not gold:
        raise ValueError("gold and predicted must have the same non-zero length")
    labels = sorted(set(gold) | set(predicted))
    per_class = {}
    f1s = []
    for label in labels:
        tp = sum(g == label and p == label for g, p in zip(gold, predicted))
        fp = sum(g != label and p == label for g, p in zip(gold, predicted))
        fn = sum(g == label and p != label for g, p in zip(gold, predicted))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1s.append(f1)
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": sum(g == label for g in gold)}
    majority = Counter(gold).most_common(1)[0][0]
    return {
        "accuracy": sum(g == p for g, p in zip(gold, predicted)) / len(gold),
        "macro_f1": sum(f1s) / len(f1s),
        "majority_label": majority,
        "majority_accuracy": sum(g == majority for g in gold) / len(gold),
        "per_class": per_class,
    }


def aurc(correct: list[bool], confidence: list[float]) -> float:
    """Area under the empirical risk-coverage curve; lower is better."""
    if len(correct) != len(confidence) or not correct:
        raise ValueError("correct and confidence must have the same non-zero length")
    ordered = sorted(zip(confidence, correct), reverse=True)
    errors = 0
    risks = []
    for index, (_, is_correct) in enumerate(ordered, 1):
        errors += int(not is_correct)
        risks.append(errors / index)
    return sum(risks) / len(risks)


def accuracy_at_coverage(correct: list[bool], confidence: list[float], coverage: float) -> float:
    if not 0 < coverage <= 1:
        raise ValueError("coverage must be in (0, 1]")
    keep = max(1, round(len(correct) * coverage))
    selected = sorted(zip(confidence, correct), reverse=True)[:keep]
    return sum(item[1] for item in selected) / len(selected)


def oracle_gap_recovery(vanilla: float, routed: float, oracle: float) -> float | None:
    gap = oracle - vanilla
    return (routed - vanilla) / gap if gap > 0 else None


def harm_prevention_rate(rows: Iterable[dict[str, object]]) -> float | None:
    harmful = [row for row in rows if bool(row["closed_book_correct"]) and not bool(row["vanilla_rag_correct"])]
    if not harmful:
        return None
    prevented = sum(str(row["route"]) != "vanilla_rag" for row in harmful)
    return prevented / len(harmful)
