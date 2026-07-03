"""Trích luật từ từng cây quyết định trong Random Forest."""
from typing import List

import numpy as np
from sklearn.tree import _tree

from src.rules.rule_types import Condition, Rule, RuleSet


class RuleExtractor:
    def extract(self, rf_model) -> RuleSet:
        all_rules: List[Rule] = []
        for tree in rf_model.estimators_:
            self._traverse_node(tree.tree_, 0, [], all_rules)
        return RuleSet(rules=all_rules)

    def _traverse_node(self, tree_struct, node_id: int, current_conditions, all_rules) -> None:
        left = tree_struct.children_left[node_id]
        right = tree_struct.children_right[node_id]
        if left == _tree.TREE_LEAF or right == _tree.TREE_LEAF:
            target_class = int(np.argmax(tree_struct.value[node_id][0]))
            all_rules.append(Rule(current_conditions.copy(), target_class))
            return
        feat, thres = tree_struct.feature[node_id], tree_struct.threshold[node_id]
        current_conditions.append(Condition(feat, "<=", thres))
        self._traverse_node(tree_struct, left, current_conditions, all_rules)
        current_conditions.pop()
        current_conditions.append(Condition(feat, ">", thres))
        self._traverse_node(tree_struct, right, current_conditions, all_rules)
        current_conditions.pop()
