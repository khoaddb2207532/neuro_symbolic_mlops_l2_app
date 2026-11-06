from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class RuleMatch(BaseModel):
    rule: str
    satisfaction_score: float
    rule_confidence: float


class BoundingBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int
    importance: float


class RuleRegion(BaseModel):
    rule_number: int
    rule: str
    satisfaction_score: float
    color: List[int]
    boxes: List[BoundingBox]
    method: str


class ClassScore(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    class_: str = Field(alias="class")
    confidence: float


class PredictionResponse(BaseModel):
    model_key: str
    model_label: str
    model_description: str
    available: bool = True
    predicted_class: str
    predicted_class_index: int
    confidence: float
    top5: List[dict]
    matched_rules: List[RuleMatch]
    rule_regions: List[RuleRegion] = Field(default_factory=list)
    annotated_image_base64: str | None = None
    inference_ms: float
    explanation_ms: float


class ModelComparisonItem(BaseModel):
    model_key: str
    model_label: str
    model_description: str
    available: bool
    error: Optional[str] = None
    expected_checkpoint: Optional[str] = None
    train_command: Optional[str] = None
    predicted_class: Optional[str] = None
    predicted_class_index: Optional[int] = None
    confidence: Optional[float] = None
    top5: List[dict] = Field(default_factory=list)
    matched_rules: List[RuleMatch] = Field(default_factory=list)
    rule_regions: List[RuleRegion] = Field(default_factory=list)
    annotated_image_base64: Optional[str] = None
    inference_ms: Optional[float] = None
    explanation_ms: Optional[float] = None


class ComparisonResponse(BaseModel):
    models: List[ModelComparisonItem]


class HealthResponse(BaseModel):
    status: str
    device: str
    num_classes: int
    num_rules: int
    models: List[dict] = Field(default_factory=list)
