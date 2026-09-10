import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

from satquery.geo.image_loader import GeoImage, ImageValidationError, load_image
from satquery.schemas.vqa import VQAResponse
from satquery.utils.logging import get_logger, record_execution
from satquery.vqa.model import BaseVQAModel, get_vqa_model

logger = get_logger("satquery.vqa.inference")


def answer_question(
    image_path: Union[str, Path],
    question: str,
    model: Optional[BaseVQAModel] = None,
    model_id: Optional[str] = None,
    force_mock: bool = False,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Core VQA API for SatQuery AI (VQA V2).
    
    Processes a satellite image and natural language question with optional evidence context,
    executes the Remote-Sensing VLM, records provenance, and returns structured results.
    """
    start_time = time.time()
    
    if not question or not str(question).strip():
        raise ValueError("Question cannot be empty.")

    clean_question = str(question).strip()
    logger.info(f"Received VQA query: '{clean_question}' on image: {image_path}")

    # 1. Load and inspect image
    geo_img: GeoImage = load_image(image_path)
    logger.info(
        f"Image loaded: shape={geo_img.shape}, geospatial={geo_img.is_geospatial}, CRS={geo_img.crs}"
    )

    # 2. Acquire model instance
    vlm_model = model or get_vqa_model(model_id=model_id, force_mock=force_mock)

    # 3. Generate answer
    logger.info(f"Running VLM inference using {vlm_model.model_id} (Mode: {vlm_model.mode}, RS Adaptation: {vlm_model.rs_adaptation_status})...")
    vqa_result = vlm_model.answer(
        image=geo_img.pil_image,
        question=clean_question,
        context=context,
    )
    answer_text = vqa_result.answer
    confidence = vqa_result.confidence

    inference_time = time.time() - start_time
    logger.info(f"Inference completed in {inference_time:.3f}s. Model: {vlm_model.model_id}")

    # 4. Record provenance / execution trace
    record_execution(
        task="vqa",
        model_name=f"{vlm_model.model_id} ({vlm_model.mode})",
        image_path=str(image_path),
        question=clean_question,
        answer=answer_text,
        execution_time_sec=inference_time,
        metadata=geo_img.metadata,
        confidence=confidence,
    )

    # 5. Build structured response
    response = VQAResponse(
        answer=answer_text,
        model=vlm_model.model_id,
        model_mode=vlm_model.mode,
        rs_adaptation=vlm_model.rs_adaptation_status,
        confidence=confidence,
        metadata={
            **geo_img.metadata,
            "mode": vlm_model.mode,
            "rs_adaptation": vlm_model.rs_adaptation_status,
            "evidence_count": len(vqa_result.evidence),
        },
        execution_time_sec=round(inference_time, 4),
    )

    return response.model_dump()
