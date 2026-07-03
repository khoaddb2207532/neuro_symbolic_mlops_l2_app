from typing import List

from pydantic import BaseModel


class RuleMatch(BaseModel):
    rule: str
    satisfaction_score: float
    rule_confidence: float


class ClassScore(BaseModel):
    class_: str
    confidence: float

    class Config:
        fields = {"class_": "class"}


class PredictionResponse(BaseModel):
    predicted_class: str
    predicted_class_index: int
    confidence: float
    top5: List[dict]
    matched_rules: List[RuleMatch]


class HealthResponse(BaseModel):
    status: str
    device: str
    num_classes: int
    num_rules: int
