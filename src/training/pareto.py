"""Pareto utilities for regularization search."""


def pareto_front(records, baseline_accuracy: float, max_accuracy_drop: float):
    feasible = [r for r in records if baseline_accuracy - r["accuracy"] <= max_accuracy_drop]
    front = []
    for candidate in feasible:
        dominated = any(
            other["bad_leaf_count"] <= candidate["bad_leaf_count"]
            and other["accuracy"] >= candidate["accuracy"]
            and (other["bad_leaf_count"] < candidate["bad_leaf_count"]
                 or other["accuracy"] > candidate["accuracy"])
            for other in feasible
        )
        if not dominated:
            front.append(candidate)
    return front


def choose_conservative_pareto(front):
    if not front:
        raise RuntimeError("no Pareto candidate satisfies the accuracy-drop budget")
    minimum_bad = min(record["bad_leaf_count"] for record in front)
    candidates = [record for record in front if record["bad_leaf_count"] == minimum_bad]
    return max(candidates, key=lambda record: record["accuracy"])
