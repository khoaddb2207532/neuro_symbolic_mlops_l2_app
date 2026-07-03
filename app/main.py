"""FastAPI service phục vụ inference cho model neuro-symbolic đã huấn luyện.

Chạy local:
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

Docs tự sinh tại http://localhost:8000/docs
"""
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.inference import InferenceService
from app.schemas import HealthResponse, PredictionResponse

app = FastAPI(
    title="Vietnamese Cultural Image Classifier — Neuro-Symbolic API",
    description="Phân loại ảnh văn hoá Việt Nam kèm giải thích bằng luật (rule-based explanation).",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # thu hẹp lại thành domain cụ thể khi lên production
    allow_methods=["*"],
    allow_headers=["*"],
)

_service: InferenceService = None


@app.on_event("startup")
def load_service() -> None:
    global _service
    _service = InferenceService(params_path="params.yaml")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    if _service is None:
        raise HTTPException(status_code=503, detail="Model chưa sẵn sàng.")
    return HealthResponse(
        status="ok",
        device=str(_service.device),
        num_classes=len(_service.class_names),
        num_rules=len(_service.rule_set),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    if _service is None:
        raise HTTPException(status_code=503, detail="Model chưa sẵn sàng.")
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File phải là ảnh (jpg/png/...).")

    image_bytes = await file.read()
    try:
        result = _service.predict(image_bytes)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=f"Không xử lý được ảnh: {exc}") from exc
    return PredictionResponse(**result)
