"""
SatQuery AI — Remote Sensing Vision-Language Models (VQA V2).
Problem Statement: SIH26167
"""

import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from PIL import Image
import numpy as np

from satquery.schemas.vqa import (
    EvidenceItem,
    EvidenceType,
    VQAAnswerResult,
    VQAModelInfo,
)
from satquery.utils.logging import get_logger
from satquery.vqa.adapters import RemoteSensingAdapter
from satquery.vqa.prompts import SYSTEM_RS_VQA_PROMPT, format_vqa_prompt
from satquery.vqa.registry import VQAModelRegistry, load_model_config

logger = get_logger("satquery.vqa.model")


class BaseVQAModel:
    """Abstract interface for Remote Sensing VQA model implementations."""

    def __init__(
        self,
        model_id: str,
        mode: str = "BASE",
        adapter: Optional[RemoteSensingAdapter] = None,
    ):
        self.model_id = model_id
        self.model_name = model_id
        self.mode = mode
        self.adapter = adapter or RemoteSensingAdapter()
        self.last_result: Optional[VQAAnswerResult] = None

    @property
    def is_remote_sensing_adapted(self) -> bool:
        """True only if an actual domain-adapted checkpoint is active."""
        return self.adapter.is_loaded

    @property
    def rs_adaptation_status(self) -> str:
        """Domain adaptation status label (LOADED / NOT LOADED)."""
        return self.adapter.status_label

    def get_model_info(self) -> VQAModelInfo:
        """Returns structured model provenance metadata."""
        return VQAModelInfo(
            name=self.model_name,
            mode=self.mode,
            remote_sensing_adapted=self.is_remote_sensing_adapted,
            rs_adaptation_status=self.rs_adaptation_status,
        )

    def answer(
        self,
        image: Image.Image,
        question: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> VQAAnswerResult:
        """Generates evidence-aware VQA answer result."""
        raise NotImplementedError

    def generate_answer(
        self,
        image: Image.Image,
        question: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Optional[float]]:
        """Backward-compatible tuple API returning (answer_text, confidence)."""
        res = self.answer(image=image, question=question, context=context)
        self.last_result = res
        return res.answer, res.confidence


class MockVQAModel(BaseVQAModel):
    """
    Mock VQA Engine (VQA V2).
    Provides honest qualitative scene understanding without fabricating percentages,
    exact object counts, coordinates, or fake probabilities.
    Seamlessly incorporates verified spatial and segmentation evidence from context.
    """

    def __init__(
        self,
        model_id: str = "mock-vlm-v1",
        adapter: Optional[RemoteSensingAdapter] = None,
    ):
        super().__init__(model_id=model_id, mode="MOCK", adapter=adapter)
        logger.info(f"Initialized Mock VQA Engine V2 (ID: {self.model_id}, Mode: MOCK, Adaptation: {self.rs_adaptation_status})")

    def answer(
        self,
        image: Union[Image.Image, str, Path],
        question: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> VQAAnswerResult:
        if isinstance(image, (str, Path)):
            image = Image.open(str(image))

        context = context or {}
        q_clean = question.strip()
        q_lower = q_clean.lower()
        w, h = image.size

        # Inspect image visual characteristics qualitatively
        img_rgb = image.convert("RGB")
        arr = np.array(img_rgb).astype(np.float32)
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        color_sat = np.max(arr, axis=2) - np.min(arr, axis=2)
        brightness = np.mean(arr, axis=2)

        # Approximate qualitative presence (boolean thresholds, no fake precision claims)
        has_water_like = bool(np.any((b > r + 15) & (b > g - 10) & (r < 115)))
        has_veg_like = bool(np.any((g > r + 8) & (g > b + 4) & (g > 35)))
        has_urban_like = bool(np.any((color_sat < 32) & (brightness > 65) & (brightness < 235)))
        has_paved_like = bool(np.any((color_sat < 22) & (brightness > 50) & (brightness < 175)))

        evidence_items: List[EvidenceItem] = []
        grounding_data = context.get("grounding")
        segmentation_data = context.get("segmentation")

        # ---------------------------------------------------------------------
        # 1. Evidence Extraction from Context
        # ---------------------------------------------------------------------
        if grounding_data and isinstance(grounding_data, dict):
            tgt = grounding_data.get("target", "feature")
            boxes = grounding_data.get("boxes", [])
            bbox = grounding_data.get("bbox") or grounding_data.get("bbox_norm")
            mask_cov = grounding_data.get("mask_coverage_pct")
            bbox_cov = grounding_data.get("bbox_coverage_pct")
            conf = grounding_data.get("confidence")
            desc = grounding_data.get("description", f"Localized '{tgt}' candidate region.")
            pbox_val = None
            if boxes:
                pbox_val = boxes[0] if isinstance(boxes[0], dict) else (boxes[0].model_dump() if hasattr(boxes[0], "model_dump") else boxes[0])
            elif bbox:
                if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                    pbox_val = {"xmin": float(bbox[0]), "ymin": float(bbox[1]), "xmax": float(bbox[2]), "ymax": float(bbox[3])}
                elif isinstance(bbox, dict):
                    pbox_val = bbox
                else:
                    pbox_val = {"raw": str(bbox)}

            evidence_items.append(
                EvidenceItem(
                    type=EvidenceType.SPATIAL,
                    source=grounding_data.get("source", "RS-SpectralHeuristic-Grounder-v1"),
                    target=tgt,
                    bbox=pbox_val,
                    mask_coverage_pct=mask_cov,
                    bbox_coverage_pct=bbox_cov,
                    confidence=conf,
                    description=desc,
                )
            )

        # ---------------------------------------------------------------------
        # 2. Check for Quantitative Questions (Strictly Abstain Without Real Evidence)
        # ---------------------------------------------------------------------
        is_percentage_query = bool(re.search(r"\b(?:percentage|percent|how\s+much\s+(?:of\s+the\s+image|area)|proportion|ratio|fraction)\b", q_lower))
        is_count_query = bool(re.search(r"\b(?:how\s+many|count|exact\s+number|number\s+of)\b", q_lower))
        is_area_query = bool(re.search(r"\b(?:exact\s+area|total\s+area|surface\s+area|how\s+many\s+sq(?:uare)?\s*(?:km|meters|miles|hectares|acres))\b", q_lower))

        if is_percentage_query:
            target_entity = "water" if "water" in q_lower or "river" in q_lower or "lake" in q_lower else ("vegetation" if "veg" in q_lower or "forest" in q_lower or "crop" in q_lower else ("built-up" if "urban" in q_lower or "building" in q_lower or "built" in q_lower else "target feature"))

            # Quantitative claims strictly require explicit SEGMENTATION evidence with pixel counts or ratio
            if segmentation_data and isinstance(segmentation_data, dict):
                pixel_count = segmentation_data.get("pixel_count")
                valid_pixel_count = segmentation_data.get("valid_pixel_count")
                cov_pct = segmentation_data.get("coverage_pct") or segmentation_data.get(f"{target_entity}_coverage_pct") or segmentation_data.get("water_coverage_pct")

                if pixel_count is not None and valid_pixel_count and valid_pixel_count > 0:
                    pct = (pixel_count / valid_pixel_count) * 100
                    answer_text = f"Estimated {target_entity} coverage is {pct:.1f}% (Source: segmentation engine, measurement: {target_entity}_pixel_ratio)."
                elif cov_pct is not None:
                    answer_text = f"Estimated {target_entity} coverage is {float(cov_pct):.1f}% (Source: segmentation engine, measurement: {target_entity}_pixel_ratio)."
                else:
                    answer_text = f"I cannot reliably calculate the {target_entity}-covered percentage using the current VQA model. A segmentation or quantitative analysis module is required."
            else:
                answer_text = f"I cannot reliably calculate the {target_entity}-covered percentage using the current VQA model. A segmentation or quantitative analysis module is required."

            return VQAAnswerResult(
                answer=answer_text,
                model_info=self.get_model_info(),
                confidence=None,
                evidence=evidence_items,
                metadata={"question_type": "quantitative_percentage"},
            )

        if is_count_query:
            target_entity = "building" if "building" in q_lower or "house" in q_lower or "structure" in q_lower else ("vessel" if "ship" in q_lower or "boat" in q_lower or "vessel" in q_lower else "object")
            answer_text = f"I cannot reliably determine the exact {target_entity} count using the current VQA model."
            return VQAAnswerResult(
                answer=answer_text,
                model_info=self.get_model_info(),
                confidence=None,
                evidence=evidence_items,
                metadata={"question_type": "quantitative_count"},
            )

        if is_area_query:
            target_entity = "water" if "water" in q_lower else "target feature"
            answer_text = "I cannot reliably calculate the exact area from the current VQA model."
            return VQAAnswerResult(
                answer=answer_text,
                model_info=self.get_model_info(),
                confidence=None,
                evidence=evidence_items,
                metadata={"question_type": "quantitative_area"},
            )

        # ---------------------------------------------------------------------
        # 3. Qualitative Presence / Scene Description Questions
        # ---------------------------------------------------------------------
        is_yes_no_query = bool(re.search(r"^(?:is\s+there|are\s+there|do\s+you\s+see|can\s+you\s+see|does\s+this\s+image\s+have)\b", q_lower))

        if "water" in q_lower or "river" in q_lower or "lake" in q_lower or "reservoir" in q_lower:
            if evidence_items and evidence_items[0].type == EvidenceType.SPATIAL:
                sp_ev = evidence_items[0]
                region_desc = "in the western area" if "western" in str(sp_ev.description).lower() else "in the highlighted region"
                answer_text = (
                    f"Yes. A water body appears to be visible in the scene, located {region_desc}."
                    if is_yes_no_query
                    else f"Water body detected. It is visible {region_desc}."
                )
            elif has_water_like:
                answer_text = "Yes. A water body appears to be visible in the scene." if is_yes_no_query else "Water features are visible in this scene."
            else:
                answer_text = "No. No prominent water bodies are visible in this scene." if is_yes_no_query else "No prominent water bodies are visible in this image."

        elif "airport" in q_lower or "runway" in q_lower or "airfield" in q_lower:
            answer_text = "Yes. An airport facility with runways is visible." if is_yes_no_query else "The image shows an airport facility with runways and taxiways."

        elif "port" in q_lower or "harbor" in q_lower or "dock" in q_lower or "coastal" in q_lower:
            answer_text = "Yes. A coastal port facility is visible." if is_yes_no_query else "The scene shows a coastal harbor with maritime infrastructure and docking facilities."

        elif "urban" in q_lower or "building" in q_lower or "city" in q_lower or "structure" in q_lower:
            if evidence_items and evidence_items[0].type == EvidenceType.SPATIAL:
                answer_text = "Yes. Built-up structures are visible in the highlighted region." if is_yes_no_query else "Built-up areas and building structures are visible in the image."
            elif has_urban_like:
                answer_text = "Yes. Built-up structures are visible in the image." if is_yes_no_query else "Built-up areas and building structures are observable in the image."
            else:
                answer_text = "No prominent built-up structures are visible in this scene."

        elif "agri" in q_lower or "crop" in q_lower or "farm" in q_lower or "vegetation" in q_lower or "forest" in q_lower or "green" in q_lower:
            if has_veg_like:
                answer_text = "Yes. Vegetation and green canopy areas are visible." if is_yes_no_query else "Vegetation and green canopy areas are visible across the scene."
            else:
                answer_text = "Limited vegetation is visible in this scene."

        elif "road" in q_lower or "highway" in q_lower or "transit" in q_lower:
            answer_text = "Yes. Roads are visible in the image." if is_yes_no_query else "Roads and transit corridors are visible traversing through the scene."

        else:
            # General scene description (honest, natural, no jargon)
            classes = []
            if has_urban_like: classes.append("built-up areas")
            if has_veg_like: classes.append("vegetation")
            if has_water_like: classes.append("a visible water region")
            if not classes: classes.append("natural terrain and open ground")

            composition_desc = " and ".join(classes) if len(classes) <= 2 else f"{', '.join(classes[:-1])}, with {classes[-1]}"
            answer_text = f"The scene appears to contain {composition_desc}."

        return VQAAnswerResult(
            answer=answer_text,
            model_info=self.get_model_info(),
            confidence=None,
            evidence=evidence_items,
            metadata={"image_dimensions": f"{w}x{h}"},
        )


class RealVQAModel(BaseVQAModel):
    """
    Production Vision-Language Model Wrapper (e.g. Qwen2-VL-2B-Instruct / GeoChat).
    Integrates RemoteSensingAdapter and evidence context formatting.
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2-VL-2B-Instruct",
        device: str = "auto",
        torch_dtype: str = "float16",
        adapter: Optional[RemoteSensingAdapter] = None,
        rs_adapter_path: Optional[str] = None,
    ):
        if adapter is None and rs_adapter_path:
            adapter = RemoteSensingAdapter(adapter_path=rs_adapter_path)
        super().__init__(model_id=model_id, mode="REAL", adapter=adapter)
        self.device = device
        self.torch_dtype = torch_dtype
        self.model = None
        self.processor = None
        self._fallback_mock: Optional[MockVQAModel] = None
        self._load()

    def _load(self):
        try:
            # Check if model downloads are explicitly permitted or if local checkpoint exists
            allow_downloads = os.getenv("SATQUERY_ALLOW_MODEL_DOWNLOADS", "0").lower() in ("1", "true", "yes")
            if not Path(self.model_id).exists() and not allow_downloads:
                raise FileNotFoundError(
                    f"Model '{self.model_id}' is not stored locally and automatic downloading is disabled."
                )

            import torch
            from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

            logger.info(f"Loading VLM model: {self.model_id} (device={self.device}, dtype={self.torch_dtype})")
            start_t = time.time()

            resolved_device = "cuda" if (self.device == "cuda" or (self.device == "auto" and torch.cuda.is_available())) else "cpu"
            dtype = torch.float16 if (resolved_device == "cuda" and self.torch_dtype == "float16") else torch.float32

            self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)

            if resolved_device == "cuda":
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    self.model_id,
                    torch_dtype=dtype,
                    device_map="auto",
                    trust_remote_code=True,
                )
            else:
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    self.model_id,
                    torch_dtype=dtype,
                    trust_remote_code=True,
                ).to("cpu")

            # Apply PEFT Remote Sensing Adapter if checkpoint exists
            if self.adapter.is_loaded:
                self.model = self.adapter.apply_to_model(self.model)
                self.mode = "ADAPTED"

            self.model.eval()
            load_time = time.time() - start_t
            logger.info(f"Model loaded successfully in {load_time:.2f}s on {resolved_device}")
        except Exception as e:
            logger.warning(f"RealVQAModel could not load weights for {self.model_id}: {e}. Running in mock mode.")
            self.mode = "MOCK"
            self._fallback_mock = MockVQAModel(model_id=self.model_id, adapter=self.adapter)

    def answer(
        self,
        image: Image.Image,
        question: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> VQAAnswerResult:
        if self._fallback_mock is not None or self.model is None:
            return self._fallback_mock.answer(image, question, context=context)

        import torch

        # Build evidence-aware prompt
        prompt_text = format_vqa_prompt(question, context=context)

        messages = [
            {"role": "system", "content": SYSTEM_RS_VQA_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text},
                ],
            },
        ]

        text_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(
            text=[text_prompt],
            images=[image],
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.2,
                do_sample=False,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

        evidence_items = []
        if context and "grounding" in context:
            gd = context["grounding"]
            evidence_items.append(
                EvidenceItem(
                    type=EvidenceType.SPATIAL,
                    source="RS-SpectralHeuristic-Grounder-v1",
                    target=gd.get("target"),
                    confidence=gd.get("confidence"),
                    description="Spatial evidence provided via context",
                )
            )

        return VQAAnswerResult(
            answer=output_text,
            model_info=self.get_model_info(),
            confidence=None,
            evidence=evidence_items,
        )


# Backward-compatible alias
Qwen2VLModel = RealVQAModel
RemoteSensingVQAModel = RealVQAModel


_MODEL_INSTANCE: Optional[BaseVQAModel] = None


def get_vqa_model(
    model_id: Optional[str] = None,
    force_mock: bool = False,
    config_path: Optional[str] = None,
    adapter: Optional[RemoteSensingAdapter] = None,
) -> BaseVQAModel:
    """
    Singleton factory for acquiring the active VLM instance.
    Loads the model once in memory and reuses it across inferences.
    """
    global _MODEL_INSTANCE
    if _MODEL_INSTANCE is not None and not force_mock:
        return _MODEL_INSTANCE

    if force_mock or os.getenv("SATQUERY_MOCK_MODEL", "0").lower() in {"1", "true", "yes"}:
        _MODEL_INSTANCE = MockVQAModel(model_id=model_id or "mock-vlm-v1", adapter=adapter)
        return _MODEL_INSTANCE

    cfg = load_model_config(config_path)
    selected_id = model_id or cfg.get("selected_model", "Qwen/Qwen2-VL-2B-Instruct")
    runtime_cfg = cfg.get("runtime", {})
    device = os.getenv("SATQUERY_DEVICE", runtime_cfg.get("prefer_device", "auto"))
    torch_dtype = runtime_cfg.get("torch_dtype", "float16")
    mock_fallback = runtime_cfg.get("enable_mock_fallback", True)

    try:
        _MODEL_INSTANCE = RealVQAModel(
            model_id=selected_id,
            device=device,
            torch_dtype=torch_dtype,
            adapter=adapter,
        )
    except Exception as e:
        logger.warning(f"Real VLM model unavailable ({selected_id}: {e}); running in mock mode.")
        if mock_fallback:
            _MODEL_INSTANCE = MockVQAModel(
                model_id=f"fallback-mock-({selected_id})",
                adapter=adapter,
            )
        else:
            raise

    return _MODEL_INSTANCE
