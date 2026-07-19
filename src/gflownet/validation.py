"""Dependency-light validation of exported class-conditional leaf tables."""


def validate_leaf_probability_coverage(probabilities, good_groups, bad_groups) -> None:
    for kind, groups in (("good", good_groups), ("bad", bad_groups)):
        table = probabilities.get(kind, {})
        for class_id, leaf_ids in groups.items():
            expected = {str(int(i)) for i in leaf_ids}
            actual = set(table.get(str(class_id), {}))
            if actual != expected:
                raise RuntimeError(f"{kind}/{class_id} probability coverage mismatch")
            if expected:
                total = sum(table[str(class_id)].values())
                if abs(total - 1.0) > 1e-6:
                    raise RuntimeError(f"{kind}/{class_id} probabilities do not sum to one")
