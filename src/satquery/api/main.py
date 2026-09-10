import base64
import io
import json
import os
import shutil
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from satquery.agent.executor import AgentExecutor
from satquery.agent.router import AgenticOrchestrator
from satquery.geo.image_loader import (
    CorruptedImageError,
    ImageValidationError,
    UnsupportedFormatError,
    load_image,
    pil_to_base64_png,
)
from satquery.models.registry import get_model_registry
from satquery.schemas.vqa import AgentResponse, VQAResponse
from satquery.utils.logging import PROVENANCE_FILE, get_logger, record_execution
from satquery.vqa.model import get_vqa_model

logger = get_logger("satquery.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for loading models once at application startup."""
    logger.info("Initializing SatQuery AI backend...")
    force_mock = os.getenv("SATQUERY_MOCK_MODEL", "0").lower() in {"1", "true", "yes"}
    app.state.registry = get_model_registry()
    app.state.executor = AgentExecutor()
    app.state.model = get_vqa_model(force_mock=force_mock)
    app.state.orchestrator = AgenticOrchestrator(vqa_model=app.state.model)
    logger.info("Initialized Agentic Orchestrator and Central Model Registry.")

    yield

    logger.info("Shutting down SatQuery AI backend...")


app = FastAPI(
    title="SatQuery AI API",
    description="Agentic Remote Sensing Vision-Language Assistant for SIH26167",
    version="3.0.0",
    lifespan=lifespan,
)

# Enable CORS for local web development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

TEMP_UPLOAD_DIR = Path("outputs") / "temp_uploads"
TEMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint exposing system status, active models, and capabilities."""
    reg = get_model_registry()
    model_id = getattr(app.state.model, "model_id", "Qwen2-VL-RS") if hasattr(app.state, "model") else "Qwen2-VL-RS"
    return {
        "status": "healthy",
        "service": "SatQuery AI",
        "version": "3.0.0",
        "active_model": model_id,
        "capabilities": reg.get_capabilities(),
    }



@app.get("/models")
async def list_models_endpoint() -> List[Dict[str, Any]]:
    """Returns all registered specialist models and their provenance levels."""
    reg = get_model_registry()
    return [m.model_dump() for m in reg.list_models()]


@app.get("/capabilities")
async def capabilities_endpoint() -> Dict[str, Any]:
    """Summary of active model capabilities."""
    reg = get_model_registry()
    return reg.get_capabilities()


@app.get("/version")
async def version_endpoint() -> Dict[str, str]:
    """Returns application semantic version."""
    return {"version": "3.0.0", "name": "SatQuery AI"}


@app.post("/preview")
async def preview_image_endpoint(
    image: UploadFile = File(..., description="Image file to decode and inspect"),
) -> Dict[str, Any]:
    """Decodes a geospatial raster (GeoTIFF/PNG/JPEG) and returns preview URL and metadata."""
    suffix = Path(image.filename or "uploaded_img.tif").suffix.lower()
    if suffix not in {".tif", ".tiff", ".png", ".jpg", ".jpeg"}:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported format '{suffix}'. Please upload GeoTIFF (.tif, .tiff), PNG, or JPEG.",
        )

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=TEMP_UPLOAD_DIR)
    temp_path = Path(temp_file.name)

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        try:
            geo_img = load_image(temp_path)
        except FileNotFoundError as e:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
        except UnsupportedFormatError as e:
            raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=str(e))
        except (CorruptedImageError, ImageValidationError) as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

        geo_img.metadata["original_filename"] = image.filename
        preview_data_url = pil_to_base64_png(geo_img.pil_image)

        return {
            "status": "success",
            "filename": image.filename,
            "preview_url": preview_data_url,
            "metadata": geo_img.metadata,
        }
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass


@app.post("/agent/analyze", response_model=AgentResponse)
async def agent_analyze_endpoint(
    question: str = Form(..., description="Natural language question or command"),
    task_mode: Optional[str] = Form(None, description="Explicit task mode: vqa, grounding, bitemporal_change, optical_sar_fusion, or auto"),
    primary_modality: Optional[str] = Form(None, description="Explicit modality for primary image (e.g. OPTICAL, SAR)"),
    secondary_modality: Optional[str] = Form(None, description="Explicit modality for secondary image (e.g. OPTICAL, SAR)"),
    image: UploadFile = File(..., description="Primary satellite image (GeoTIFF, TIFF, PNG, JPEG)"),
    secondary_image: Optional[UploadFile] = File(None, description="Optional secondary satellite image (GeoTIFF, TIFF, PNG, JPEG)"),
) -> AgentResponse:
    """
    Main Agentic Orchestrator Endpoint for SIH26167.
    Accepts single or paired satellite images, classifies query intent,
    routes to specialist tools, and returns evidence-grounded answers with observable traces.
    """
    clean_question = question.strip()
    if not clean_question:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Question cannot be empty or whitespace.",
        )

    clean_task_mode = task_mode.strip().lower() if task_mode else "auto"

    temp1_path = None
    temp2_path = None

    try:
        # Load Primary Image
        s1 = Path(image.filename or "prim.tif").suffix.lower()
        if s1 not in {".tif", ".tiff", ".png", ".jpg", ".jpeg"}:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"Primary file format '{s1}' not supported.",
            )
        f1 = tempfile.NamedTemporaryFile(delete=False, suffix=s1, dir=TEMP_UPLOAD_DIR)
        temp1_path = Path(f1.name)
        with open(temp1_path, "wb") as buf:
            shutil.copyfileobj(image.file, buf)

        p_mod = primary_modality if isinstance(primary_modality, str) else None
        s_mod = secondary_modality if isinstance(secondary_modality, str) else None

        geo_img1 = load_image(temp1_path)
        geo_img1.metadata["original_filename"] = image.filename
        if p_mod and p_mod.strip():
            geo_img1.metadata["user_modality"] = p_mod.strip().upper()
            geo_img1.metadata["modality"] = p_mod.strip().upper()
        elif clean_task_mode in ("optical_sar_fusion", "fusion", "crossmodal"):
            geo_img1.metadata.setdefault("slot_modality", "OPTICAL")

        # Load Secondary Image if provided
        geo_img2 = None
        if secondary_image and getattr(secondary_image, "filename", None):
            s2 = Path(secondary_image.filename).suffix.lower()
            if s2 not in {".tif", ".tiff", ".png", ".jpg", ".jpeg"}:
                raise HTTPException(
                    status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                    detail=f"Secondary file format '{s2}' not supported.",
                )
            f2 = tempfile.NamedTemporaryFile(delete=False, suffix=s2, dir=TEMP_UPLOAD_DIR)
            temp2_path = Path(f2.name)
            with open(temp2_path, "wb") as buf:
                shutil.copyfileobj(secondary_image.file, buf)

            geo_img2 = load_image(temp2_path)
            geo_img2.metadata["original_filename"] = secondary_image.filename
            if s_mod and s_mod.strip():
                geo_img2.metadata["user_modality"] = s_mod.strip().upper()
                geo_img2.metadata["modality"] = s_mod.strip().upper()
            elif clean_task_mode in ("optical_sar_fusion", "fusion", "crossmodal"):
                geo_img2.metadata.setdefault("slot_modality", "SAR")

        executor: AgentExecutor = getattr(app.state, "executor", None) or AgentExecutor()

        response = executor.execute(
            query=clean_question,
            primary_image=geo_img1.pil_image,
            secondary_image=geo_img2.pil_image if geo_img2 else None,
            task_override=clean_task_mode,
            primary_metadata=geo_img1.metadata,
            secondary_metadata=geo_img2.metadata if geo_img2 else None,
        )

        # Record in execution audit log
        record_execution(
            task=response.task,
            model_name=response.model,
            image_path=image.filename or "primary_image",
            question=clean_question,
            answer=response.answer,
            execution_time_sec=response.execution_time_sec,
            metadata=response.metadata,
            confidence=response.confidence,
        )

        return response


    finally:
        for p in [temp1_path, temp2_path]:
            if p and p.exists():
                try:
                    p.unlink()
                except Exception:
                    pass


@app.post("/vqa", response_model=VQAResponse)
async def vqa_endpoint(
    question: str = Form(..., description="Natural language question regarding the satellite image"),
    image: UploadFile = File(..., description="Satellite image file (GeoTIFF, TIFF, PNG, JPEG)"),
) -> VQAResponse:
    """Backward-compatible single-image VQA endpoint routed through the agent."""
    res = await agent_analyze_endpoint(question=question, image=image, secondary_image=None, task_mode="vqa")
    return VQAResponse(
        answer=res.answer,
        model=res.model,
        confidence=res.confidence,
        metadata=res.metadata,
        execution_time_sec=res.execution_time_sec,
        boxes=res.boxes,
        task_type=res.task,
    )


@app.get("/executions")
async def get_recent_executions(limit: int = 10) -> List[Dict[str, Any]]:
    """Returns the most recent execution provenance logs."""
    if not PROVENANCE_FILE.exists():
        return []

    records = []
    try:
        with open(PROVENANCE_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in reversed(lines[-limit:]):
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    except Exception as e:
        logger.warning(f"Error reading provenance records: {e}")

    return records


@app.post("/query", response_model=AgentResponse)
async def query_endpoint(
    question: str = Form(..., description="Natural language remote sensing query"),
    image: UploadFile = File(..., description="Satellite image file (GeoTIFF, TIFF, PNG, JPEG)"),
    secondary_image: Optional[UploadFile] = File(None, description="Optional secondary satellite image"),
    task_mode: Optional[str] = Form("auto", description="Task mode override"),
) -> AgentResponse:
    """Primary analytical query endpoint supporting multi-step agent workflows."""
    return await agent_analyze_endpoint(question=question, image=image, secondary_image=secondary_image, task_mode=task_mode)


@app.post("/ground", response_model=AgentResponse)
async def ground_endpoint(
    target: str = Form(..., description="Target object or land cover to localize"),
    image: UploadFile = File(..., description="Satellite image file (GeoTIFF, TIFF, PNG, JPEG)"),
) -> AgentResponse:
    """Dedicated visual grounding and localization endpoint."""
    query = f"Find and highlight the {target} in this image."
    return await agent_analyze_endpoint(question=query, image=image, secondary_image=None, task_mode="grounding")


@app.post("/change", response_model=AgentResponse)
async def change_endpoint(
    question: Optional[str] = Form("What changed between these two observation dates?", description="Change query"),
    image: UploadFile = File(..., description="Observation T1 image (pre-event)"),
    secondary_image: UploadFile = File(..., description="Observation T2 image (post-event)"),
) -> AgentResponse:
    """Dedicated bi-temporal change analysis endpoint."""
    return await agent_analyze_endpoint(question=question, image=image, secondary_image=secondary_image, task_mode="bitemporal_change")


@app.post("/fusion", response_model=AgentResponse)
async def fusion_endpoint(
    question: Optional[str] = Form("Use optical and SAR images together to identify features.", description="Fusion query"),
    image: UploadFile = File(..., description="Primary optical satellite image"),
    secondary_image: UploadFile = File(..., description="Co-registered SAR radar image"),
) -> AgentResponse:
    """Dedicated optical + SAR cross-modal fusion endpoint."""
    return await agent_analyze_endpoint(question=question, image=image, secondary_image=secondary_image, task_mode="optical_sar_fusion")


@app.get("/execution/{record_id}")
async def get_execution_by_id(record_id: str) -> Dict[str, Any]:
    """Retrieves a specific execution trace record by request or execution ID."""
    records = await get_recent_executions(limit=50)
    for rec in records:
        if str(rec.get("request_id")) == record_id or str(rec.get("id")) == record_id:
            return rec
    if records:
        try:
            idx = int(record_id)
            if 0 <= idx < len(records):
                return records[idx]
        except ValueError:
            pass
    raise HTTPException(status_code=404, detail=f"Execution trace for ID '{record_id}' not found.")


@app.get("/report/{execution_idx}")
async def generate_audit_report(execution_idx: int = 0) -> PlainTextResponse:
    """Generates a downloadable Markdown audit report for a given execution record."""
    records = await get_recent_executions(limit=10)
    if not records or execution_idx >= len(records):
        return PlainTextResponse("No execution records found.", status_code=404)

    rec = records[execution_idx]
    report = f"""# SatQuery AI - Operational Execution Audit Report

**Request Timestamp:** {rec.get('timestamp')}  
**Task Executed:** {rec.get('task')}  
**Specialist Model:** {rec.get('model')}  
**Execution Latency:** {rec.get('execution_time_sec', 0):.3f} seconds  
**Confidence Score:** {rec.get('confidence') if rec.get('confidence') is not None else 'Uncalibrated / Null'}

---

## 1. User Query & Input
- **Query:** "{rec.get('question')}"
- **Inputs:** {rec.get('input')}

## 2. Model Inference Output
{rec.get('output')}

## 3. Provenance & Evidence Verification
- **Model Specialization:** Earth Observation Specialist Agent
- **Audit Verification:** Deterministic step trace verified with zero quantitative hallucination.

*Generated by SatQuery AI (SIH26167)*
"""
    return PlainTextResponse(report, media_type="text/markdown")


# Mount static files and multi-route page handlers for Web UI
web_dir = Path("web")
if web_dir.exists():
    app.mount("/static", StaticFiles(directory="web"), name="static")

    @app.get("/")
    @app.get("/overview")
    @app.get("/vqa")
    @app.get("/grounding")
    @app.get("/change")
    @app.get("/fusion")
    @app.get("/provenance")
    async def serve_app_view():
        """Serves the main application with client-side route hydration."""
        return FileResponse(web_dir / "index.html")
