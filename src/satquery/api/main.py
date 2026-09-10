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
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from satquery.agent.router import AgenticOrchestrator
from satquery.geo.image_loader import (
    CorruptedImageError,
    ImageValidationError,
    UnsupportedFormatError,
    load_image,
)
from satquery.schemas.vqa import AgentResponse, VQAResponse
from satquery.utils.logging import PROVENANCE_FILE, get_logger, record_execution
from satquery.vqa.model import BaseVQAModel, get_vqa_model

logger = get_logger("satquery.api")


def pil_to_base64_png(img) -> str:
    """Encodes a PIL image to a Base64 PNG data URL."""
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode("utf-8")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for loading models once at application startup."""
    logger.info("Initializing SatQuery AI backend...")
    force_mock = os.getenv("SATQUERY_MOCK_MODEL", "0").lower() in {"1", "true", "yes"}
    try:
        # Pre-load the active VLM singleton
        app.state.model = get_vqa_model(force_mock=force_mock)
        logger.info(f"Loaded active model: {app.state.model.model_id}")
    except Exception as e:
        logger.error(f"Error initializing model on startup: {e}")
        # Fallback to mock for resilient startup
        app.state.model = get_vqa_model(force_mock=True)

    # Initialize the Agentic Orchestrator
    app.state.orchestrator = AgenticOrchestrator(vqa_model=app.state.model)
    logger.info("Initialized Agentic Orchestrator with specialist tools.")

    yield

    logger.info("Shutting down SatQuery AI backend...")


app = FastAPI(
    title="SatQuery AI API",
    description="Agentic Remote Sensing Vision-Language Assistant for SIH26167",
    version="0.2.0",
    lifespan=lifespan,
)

# Enable CORS for local web development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_UPLOAD_DIR = Path("outputs") / "temp_uploads"
TEMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint exposing system status and active model."""
    model_id = getattr(app.state.model, "model_id", "uninitialized") if hasattr(app.state, "model") else "none"
    return {
        "status": "healthy",
        "service": "SatQuery AI",
        "version": "0.2.0",
        "active_model": model_id,
        "agentic_capabilities": [
            "single_image_vqa",
            "visual_grounding",
            "bitemporal_change_detection",
            "optical_sar_crossmodal_fusion",
            "geospatial_provenance_audit",
        ],
    }


@app.post("/preview")
async def preview_image_endpoint(
    image: UploadFile = File(..., description="Satellite image file to convert for browser preview"),
) -> Dict[str, Any]:
    """
    Accepts any supported satellite format (GeoTIFF, TIFF, PNG, JPG),
    normalizes it via the geospatial engine (percentile stretch & multi-band reduction),
    and returns a base64 encoded PNG for instant browser rendering alongside geospatial metadata.
    """
    suffix = Path(image.filename).suffix if image.filename else ".tif"
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

    # Clean explicit task_mode
    clean_task_mode = task_mode if isinstance(task_mode, str) and task_mode.lower() not in ("auto", "none", "") else None

    # Save primary image
    image.file.seek(0)
    suffix1 = Path(image.filename).suffix if image.filename else ".tif"
    temp1 = tempfile.NamedTemporaryFile(delete=False, suffix=suffix1, dir=TEMP_UPLOAD_DIR)
    temp1_path = Path(temp1.name)

    temp2_path = None
    try:
        with open(temp1_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        try:
            geo_img1 = load_image(temp1_path)
        except (UnsupportedFormatError, CorruptedImageError, ImageValidationError) as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Primary image error: {str(e)}")

        geo_img1.metadata["original_filename"] = image.filename

        geo_img2 = None
        if secondary_image is not None and getattr(secondary_image, "filename", None):
            secondary_image.file.seek(0)
            suffix2 = Path(secondary_image.filename).suffix or ".tif"
            temp2 = tempfile.NamedTemporaryFile(delete=False, suffix=suffix2, dir=TEMP_UPLOAD_DIR)
            temp2_path = Path(temp2.name)
            with open(temp2_path, "wb") as buffer:
                shutil.copyfileobj(secondary_image.file, buffer)
            try:
                geo_img2 = load_image(temp2_path)
                geo_img2.metadata["original_filename"] = secondary_image.filename
            except (UnsupportedFormatError, CorruptedImageError, ImageValidationError) as e:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Secondary image error: {str(e)}")

        # Acquire or initialize Agentic Orchestrator
        orchestrator: AgenticOrchestrator = getattr(app.state, "orchestrator", None)
        if orchestrator is None:
            active_model = getattr(app.state, "model", None)
            orchestrator = AgenticOrchestrator(vqa_model=active_model)
            app.state.orchestrator = orchestrator

        response = orchestrator.execute(
            query=clean_question,
            primary_image=geo_img1.pil_image,
            secondary_image=geo_img2.pil_image if geo_img2 else None,
            primary_meta=geo_img1.metadata,
            secondary_meta=geo_img2.metadata if geo_img2 else None,
            explicit_task=clean_task_mode,
            primary_filename=image.filename or "primary_image",
            secondary_filename=secondary_image.filename if secondary_image and getattr(secondary_image, "filename", None) else None,
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
    """
    Backward-compatible single-image VQA endpoint routed through the agent.
    """
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

