from src.training.pareto import choose_conservative_pareto, pareto_front


def test_pareto_respects_predeclared_accuracy_budget():
    records = [
        {"name": "safe", "accuracy": 0.90, "bad_leaf_count": 8},
        {"name": "strong", "accuracy": 0.895, "bad_leaf_count": 4},
        {"name": "too_destructive", "accuracy": 0.85, "bad_leaf_count": 1},
        {"name": "dominated", "accuracy": 0.89, "bad_leaf_count": 9},
    ]
    front = pareto_front(records, baseline_accuracy=0.90, max_accuracy_drop=0.01)
    assert {record["name"] for record in front} == {"safe", "strong"}
    assert choose_conservative_pareto(front)["name"] == "strong"
