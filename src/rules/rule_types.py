"""Kiểu dữ liệu cho luật (rule) trích từ Random Forest."""
from dataclasses import dataclass, field
from typing import List


@dataclass
class Condition:
    feature_index: int
    operator: str  # '<=' hoặc '>'
    threshold: float

    def __str__(self) -> str:
        return f"F_{self.feature_index} {self.operator} {self.threshold:.3f}"


@dataclass
class Rule:
    conditions: List[Condition]
    target_class: int
    confidence: float = 1.0

    def __str__(self) -> str:
        return " AND ".join(str(c) for c in self.conditions) + f" => Class {self.target_class}"


@dataclass
class RuleSet:
    rules: List[Rule] = field(default_factory=list)

    def add_rule(self, rule: Rule) -> None:
        self.rules.append(rule)

    def __len__(self) -> int:
        return len(self.rules)

    def __iter__(self):
        return iter(self.rules)

    def filter_rules(self, indices: List[int]) -> "RuleSet":
        selected = [self.rules[i] for i in indices if 0 <= i < len(self.rules)]
        return RuleSet(rules=selected)
