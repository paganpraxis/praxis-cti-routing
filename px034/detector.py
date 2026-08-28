from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass


@dataclass
class CentroidDetector:
    """Deterministic, dependency-free structural baseline.

    This is a baseline, not the intended final classifier. It standardizes features
    on development data and predicts the nearest per-class centroid.
    """

    means: dict[str, float]
    scales: dict[str, float]
    centroids: dict[str, dict[str, float]]
    priors: dict[str, float]

    @classmethod
    def fit(cls, rows: list[dict[str, float]], labels: list[str]) -> "CentroidDetector":
        if not rows or len(rows) != len(labels):
            raise ValueError("features and labels must have the same non-zero length")
        names = sorted(set().union(*(row.keys() for row in rows)))
        means = {name: sum(row.get(name, 0.0) for row in rows) / len(rows) for name in names}
        scales = {}
        for name in names:
            variance = sum((row.get(name, 0.0) - means[name]) ** 2 for row in rows) / len(rows)
            scales[name] = math.sqrt(variance) or 1.0
        grouped: dict[str, list[dict[str, float]]] = defaultdict(list)
        for row, label in zip(rows, labels):
            grouped[label].append(row)
        centroids = {
            label: {
                name: sum((row.get(name, 0.0) - means[name]) / scales[name] for row in members) / len(members)
                for name in names
            }
            for label, members in grouped.items()
        }
        counts = Counter(labels)
        return cls(means, scales, centroids, {label: count / len(labels) for label, count in counts.items()})

    def predict_one(self, row: dict[str, float]) -> tuple[str, float]:
        distances = {
            label: math.sqrt(
                sum((((row.get(name, 0.0) - self.means[name]) / self.scales[name]) - value) ** 2 for name, value in centroid.items())
            )
            for label, centroid in self.centroids.items()
        }
        ranked = sorted(distances.items(), key=lambda item: (item[1], -self.priors[item[0]], item[0]))
        label, best = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else best + 1.0
        confidence = 1.0 / (1.0 + math.exp(-(second - best)))
        return label, confidence
