"""FastAPI entry point for notebook-generated CBM model artifacts.

Run with: uvicorn app_cbm.main:app --host 0.0.0.0 --port 8000
"""
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app_cbm.inference import CBMInferenceService
from app_cbm.schemas import ModelCatalog, PredictionResponse

app = FastAPI(
    title="Vietnamese Cultural Image Classifier — CBM API",
    description=(
        "So sánh Baseline, CBM và CBM regularized từ artifacts sinh bởi "
        "generated_cbm_notebooks."
    ),
    version="1.0.0",
)

_service: Optional[CBMInferenceService] = None


@app.on_event("startup")
def load_service() -> None:
    global _service
    _service = CBMInferenceService()


def _get_service() -> CBMInferenceService:
    if _service is None:
        raise HTTPException(status_code=503, detail="Inference service chưa sẵn sàng.")
    return _service


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/health")
def health() -> dict:
    service = _get_service()
    catalog = service.catalog()
    available = sum(
        model["available"]
        for run in catalog["runs"]
        for model in run["models"]
    )
    return {
        "status": "ok" if available else "no_models",
        "device": str(service.device),
        "available_models": available,
        "artifact_root": catalog["artifact_root"],
    }


@app.get("/models", response_model=ModelCatalog)
def list_models() -> dict:
    """Rescan the artifact directory and list loadable models by seed/backbone."""
    return _get_service().catalog()


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    run_id: Optional[str] = Form(None),
    model_ids: Optional[str] = Form(None),
    mode: Literal["normal", "edited"] = Form("normal"),
) -> dict:
    """Predict with selected model IDs, or all available models in the selected run."""
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File phải là ảnh (jpg/png/...).")
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="File ảnh rỗng.")
    selected_ids = (
        [model_id.strip() for model_id in model_ids.split(",") if model_id.strip()]
        if model_ids else None
    )
    try:
        results = _get_service().predict(
            image_bytes,
            model_ids=selected_ids,
            run_id=run_id,
            include_localization=mode == "edited",
        )
    except LookupError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"predictions": results}
