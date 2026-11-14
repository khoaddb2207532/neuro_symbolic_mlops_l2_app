from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class ModelStatus(BaseModel):
    model_id: str
    model_type: str
    model_label: str
    available: bool
    checkpoint: Optional[str] = None
    error: Optional[str] = None


class ModelRun(BaseModel):
    run_id: str
    architecture: str
    seed: int
    models: List[ModelStatus]


class ModelCatalog(BaseModel):
    artifact_root: str
    runs: List[ModelRun]


class BoundingBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int
    importance: float


class ConceptScore(BaseModel):
    name: str
    probability: float
    activated: bool
    color: Optional[List[int]] = None
    boxes: List[BoundingBox] = Field(default_factory=list)


class ClassScore(BaseModel):
    label: str
    confidence: float


class RuleCondition(BaseModel):
    concept_name: str
    operator: Literal["<=", ">"]
    threshold: float
    value: float


class RuleExplanation(BaseModel):
    rule_number: int
    target_class: str
    confidence: float
    conditions: List[RuleCondition]


class Prediction(BaseModel):
    model_id: str
    model_type: str
    model_label: str
    architecture: str
    seed: int
    predicted_class: str
    predicted_class_index: int
    confidence: float
    top5: List[ClassScore]
    concepts: List[ConceptScore] = Field(default_factory=list)
    activated_concepts: List[ConceptScore] = Field(default_factory=list)
    rule_explanation_available: bool = False
    rule_explanation_error: Optional[str] = None
    explanation_rules: List[RuleExplanation] = Field(default_factory=list)
    concept_activation_threshold: Optional[float] = None
    annotated_image_base64: Optional[str] = None
    occlusion_method: Optional[str] = None
    occlusion_grid_size: Optional[int] = None
    inference_ms: float


class PredictionResponse(BaseModel):
    predictions: List[Prediction]
