from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from typing import Any


def stratified_split(
    rows: list[dict[str, Any]],
    seed: int,
    proportions: tuple[float, float, float] = (0.6, 0.2, 0.2),
) -> dict[str, list[str]]:
    if abs(sum(proportions) - 1.0) > 1e-9:
        raise ValueError("split proportions must sum to 1")
    grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["task"]), str(row["support_state"]))].append(str(row["query_id"]))
    result = {"train": [], "development": [], "heldout": []}
    rng = random.Random(seed)
    for key in sorted(grouped):
        ids = sorted(grouped[key])
        rng.shuffle(ids)
        n = len(ids)
        train_end = round(n * proportions[0])
        dev_end = train_end + round(n * proportions[1])
        result["train"].extend(ids[:train_end])
        result["development"].extend(ids[train_end:dev_end])
        result["heldout"].extend(ids[dev_end:])
    return {key: sorted(value) for key, value in result.items()}


def grouped_stratified_split(
    rows: list[dict[str, Any]],
    seed: int,
    group_field: str = "source_cluster_id",
    proportions: tuple[float, float, float] = (0.6, 0.2, 0.2),
) -> dict[str, list[str]]:
    """Keep source-connected queries together while approximating task/label strata."""
    if abs(sum(proportions) - 1.0) > 1e-9:
        raise ValueError("split proportions must sum to 1")
    names = ("train", "development", "heldout")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(group_field) or f"query:{row['query_id']}")].append(row)
    totals: dict[tuple[str, str], int] = defaultdict(int)
    for row in rows:
        totals[(str(row["task"]), str(row["support_state"]))] += 1
    targets = {
        name: {key: total * proportions[index] for key, total in totals.items()}
        for index, name in enumerate(names)
    }
    assigned: dict[str, list[str]] = {name: [] for name in names}
    counts: dict[str, dict[tuple[str, str], int]] = {name: defaultdict(int) for name in names}
    rng = random.Random(seed)
    ordered_groups = sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))
    # Seeded tie order without making results dependent on dict insertion order.
    ties = {group_id: rng.random() for group_id, _ in ordered_groups}
    ordered_groups.sort(key=lambda item: (-len(item[1]), ties[item[0]], item[0]))
    for _, members in ordered_groups:
        strata = Counter((str(row["task"]), str(row["support_state"])) for row in members)
        scored = []
        for name in names:
            cost = sum(
                (counts[name][key] + increment - targets[name][key]) ** 2
                - (counts[name][key] - targets[name][key]) ** 2
                for key, increment in strata.items()
            )
            size_ratio = (len(assigned[name]) + len(members)) / max(1.0, len(rows) * proportions[names.index(name)])
            scored.append((cost, size_ratio, name))
        chosen = min(scored)[2]
        assigned[chosen].extend(str(row["query_id"]) for row in members)
        for key, increment in strata.items():
            counts[chosen][key] += increment
    return {name: sorted(values) for name, values in assigned.items()}


def freeze_hash(value: Any) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()
