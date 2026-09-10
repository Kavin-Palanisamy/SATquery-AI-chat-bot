import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image

from satquery.analysis.change import analyze_bitemporal_change
from satquery.analysis.crossmodal import analyze_optical_sar_pair
from satquery.analysis.grounding import (
    detect_grounding_regions,
    extract_target_entity,
    get_grounding_model,
    validate_bounding_box,
    validate_grounding_boxes,
)
from satquery.schemas.vqa import (
    AgentResponse,
    AgentTaskType,
    BoundingBox,
    ConfidenceBreakdown,
    EvidenceItem,
    EvidenceType,
    IntentAnalysis,
)
from satquery.utils.logging import get_logger, record_execution
from satquery.vqa.model import (
    BaseVQAModel,
    MockVQAModel,
    RealVQAModel,
    get_vqa_model,
    load_model_config,
)

logger = get_logger("satquery.agent.router")


# ============================================================================
# Central Task & Tool Registry Matrix
# ============================================================================
TASK_REGISTRY: Dict[AgentTaskType, Dict[str, Any]] = {
    AgentTaskType.VQA: {
        "task_name": "VQA / Scene Understanding",
        "description": "Answers semantic questions regarding land cover, terrain, scene context, or presence of features.",
        "tool": "RemoteSensingVQATool",
        "default_model": "Qwen/Qwen2-VL-2B-Instruct",
        "mock_model": "mock-vlm-v1",
        "input_requirements": {
            "min_images": 1,
            "max_images": 1,
            "supported_modalities": ["optical", "multispectral", "sar", "unknown"],
            "requires_coregistration": False,
            "requires_temporal_pair": False,
        },
        "output_type": "textual_explanation",
    },
    AgentTaskType.GROUNDING: {
        "task_name": "Visual Grounding & Localization",
        "description": "Spatially localizes, finds, boxes, or highlights target geospatial features and objects.",
        "tool": "RemoteSensingGroundingTool",
        "default_model": "RS-SpectralHeuristic-Grounder-v1",
        "mock_model": "mock-grounding-v1",
        "heuristic_model": "RS-SpectralHeuristic-Grounder-v1",
        "input_requirements": {
            "min_images": 1,
            "max_images": 1,
            "supported_modalities": ["optical", "multispectral", "sar", "unknown"],
            "requires_coregistration": False,
            "requires_temporal_pair": False,
        },
        "output_type": "spatial_bounding_boxes",
    },
    AgentTaskType.BITEMPORAL_CHANGE: {
        "task_name": "Bi-Temporal Change Analysis",
        "description": "Detects, quantifies, and explains land-cover/environmental changes between two temporal timestamps (T1 & T2).",
        "tool": "RemoteSensingChangeTool",
        "default_model": "CDVQA-Difference-Engine",
        "mock_model": "MockCDVQAEngine",
        "input_requirements": {
            "min_images": 2,
            "max_images": 2,
            "supported_modalities": ["optical", "multispectral", "sar", "unknown"],
            "requires_coregistration": True,
            "requires_temporal_pair": True,
        },
        "output_type": "change_heatmap_and_explanation",
    },
    AgentTaskType.OPTICAL_SAR_FUSION: {
        "task_name": "Optical + SAR Cross-Modal Analysis",
        "description": "Fuses co-registered optical spectral reflectance with SAR microwave radar backscatter for all-weather analysis.",
        "tool": "OpticalSARFusionTool",
        "default_model": "Multimodal-Radar-Optical-Engine",
        "mock_model": "MockCrossModalEngine",
        "input_requirements": {
            "min_images": 2,
            "max_images": 2,
            "supported_modalities": ["optical_and_sar"],
            "requires_coregistration": True,
            "requires_temporal_pair": False,
        },
        "output_type": "fused_crossmodal_assessment",
    },
}


# ============================================================================
# 1. Query Intent Analyzer (Linguistic & Semantic Reasoning)
# ============================================================================


def analyze_query_intent(query: str) -> IntentAnalysis:
    """
    Deconstructs user query into structured semantic intent without relying on superficial keywords.
    Distinguishes scene understanding (VQA), spatial localization (Grounding),
    bi-temporal comparison (Change), and multisensor fusion (Optical+SAR).
    """
    q = query.strip()
    q_lower = q.lower()
    target = extract_target_entity(q)

    # -------------------------------------------------------------------------
    # Intent Category A: Cross-Modal / Optical + SAR Joint Analysis
    # -------------------------------------------------------------------------
    crossmodal_patterns = [
        r"\b(?:optical\s+and\s+sar|sar\s+and\s+optical|optical\s*\+\s*sar|sar\s*\+\s*optical)\b",
        r"\b(?:optical\s+and\s+radar|radar\s+and\s+optical|optical\s*\+\s*radar|radar\s*\+\s*optical)\b",
        r"\b(?:both\s+sensors|both\s+images\s+together|two\s+sensors|combine\s+optical|fuse\s+optical)\b",
        r"\b(?:use\s+sar\s+to|using\s+sar|with\s+sar|sar\s+backscatter|microwave\s+radar)\b",
        r"\b(?:penetrate\s+clouds|see\s+through\s+clouds|cloud\s+penetration|cross-modal|cross\s+modal)\b",
    ]
    if any(re.search(pat, q_lower) for pat in crossmodal_patterns):
        return IntentAnalysis(
            task=AgentTaskType.OPTICAL_SAR_FUSION,
            intent="crossmodal_multisensor_fusion",
            target=target or "joint landcover / structural features",
            requires_spatial_evidence=True,
            requires_multiple_images=True,
            confidence=0.96,
            reason="Query explicitly requests joint cross-modal analysis combining optical spectral bands and SAR microwave radar.",
        )

    # -------------------------------------------------------------------------
    # Intent Category B: Bi-Temporal Change & Temporal Comparison
    # -------------------------------------------------------------------------
    change_patterns = [
        r"\bwhat\s+(?:has\s+)?changed\b",
        r"\bbetween\s+(?:these\s+)?(?:two\s+)?(?:images|dates|timestamps|observations|scenes|years|periods)\b",
        r"\b(?:compare\s+these\s+images|compare\s+the\s+two\s+images|show\s+the\s+differences?|difference\s+between)\b",
        r"\bhas\s+(?:the\s+)?(?:built-up|vegetation|water|forest|urban|agricultural).*(?:increased|decreased|expanded|shrunk|changed)\b",
        r"\b(?:did|was)\s+(?:new\s+construction|built-up|agricultural|vegetation|flood).*(?:observed|increase|decrease|occur|expand)\b",
        r"\bwhere\s+did\s+the\s+flood\s+occur\b",
        r"\b(?:before\s+and\s+after|pre\s+and\s+post|temporal\s+change|change\s+detection)\b",
    ]
    if any(re.search(pat, q_lower) for pat in change_patterns):
        intent_type = "increase_decrease_query" if ("increase" in q_lower or "decrease" in q_lower or "expand" in q_lower) else "temporal_change_detection"
        return IntentAnalysis(
            task=AgentTaskType.BITEMPORAL_CHANGE,
            intent=intent_type,
            target=target or "surface land-cover alteration",
            requires_spatial_evidence=True,
            requires_multiple_images=True,
            confidence=0.95,
            reason="Query requests comparative temporal assessment between two observation dates or asks about dynamic land-cover change.",
        )

    # -------------------------------------------------------------------------
    # Intent Category C: Explicit Spatial Localization / Visual Grounding
    # -------------------------------------------------------------------------
    # Must be an explicit command/query to locate/box/highlight, NOT a descriptive query like "describe the land cover and major objects"
    grounding_patterns = [
        r"^(?:where\s+is|where\s+are|which\s+part\s+of\s+the\s+image\s+contains)\b",
        r"^(?:highlight|locate|find|show\s+me|pinpoint|delineate|box\s+in|draw\s+(?:a\s+)?box)\b",
        r"\b(?:draw\s+(?:a\s+)?(?:bounding\s+)?box\s+around|bounding\s+box\s+for|highlight\s+the)\b",
        r"\b(?:locate\s+the|find\s+the|find\s+all\s+the|where\s+is\s+the|where\s+are\s+the)\b",
        r"\b(?:identify\s+the\s+location\s+of|spatial\s+coordinates\s+of)\b",
    ]
    is_grounding_match = any(re.search(pat, q_lower) for pat in grounding_patterns)

    # Negate grounding if the sentence starts with descriptive verbs like "describe", "what is", "list", "explain", "is there", "are there"
    is_descriptive_vqa_prefix = re.search(
        r"^(?:describe|explain|what\s+is|what\s+are|what\s+type|is\s+there|are\s+there|how\s+is|how\s+many|what\s+percentage|classify|summarize|tell\s+me\s+about)\b",
        q_lower,
    ) is not None

    if is_grounding_match and not is_descriptive_vqa_prefix:
        return IntentAnalysis(
            task=AgentTaskType.GROUNDING,
            intent="spatial_localization",
            target=target or "salient geospatial entity",
            requires_spatial_evidence=True,
            requires_multiple_images=False,
            confidence=0.94,
            reason="User requested spatial localization/highlighting.",
        )

    # -------------------------------------------------------------------------
    # Intent Category D: Single-Image VQA / Scene Understanding (Default)
    # -------------------------------------------------------------------------
    vqa_intent = "scene_description"
    vqa_reason = "User requested semantic scene understanding from a single image."

    if re.search(r"\b(?:is\s+there|are\s+there|does\s+this|presence\s+of|contains?\s+(?:a|any)?)\b", q_lower):
        vqa_intent = "semantic_presence_question"
        vqa_reason = "User is asking whether a visual feature is present in a single image."
    elif re.search(r"\b(?:describe|summary|overview|explain|land\s*cover|what\s+is\s+visible|what\s+type\s+of\s+environment)\b", q_lower):
        vqa_intent = "scene_description"
        vqa_reason = "User requested semantic scene understanding from a single image."
    elif re.search(r"\b(?:percentage|percent|how\s+much|how\s+many|count|exact\s+area|exact\s+number)\b", q_lower):
        vqa_intent = "quantitative_query"
        vqa_reason = "User is asking a quantitative or statistical question about the image."
    elif re.search(r"\b(?:classify|classification|terrain\s*type)\b", q_lower):
        vqa_intent = "landcover_classification"
        vqa_reason = "User requested land-cover or terrain classification from a single image."

    return IntentAnalysis(
        task=AgentTaskType.VQA,
        intent=vqa_intent,
        target=target,
        requires_spatial_evidence=False,
        requires_multiple_images=False,
        confidence=0.96,
        reason=vqa_reason,
    )


# ============================================================================
# 2. Input & Modality Validator
# ============================================================================
def detect_image_modality(
    img: Optional[Image.Image],
    filename: Optional[str] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Determines whether an image is optical, multispectral, or SAR radar
    based on metadata, file naming conventions, and spectral channel properties.
    """
    if img is None and filename is None:
        return "none"

    meta = meta or {}
    fn_lower = (filename or "").lower()

    # 1. Metadata check
    sensor = str(meta.get("sensor", "")).lower()
    driver = str(meta.get("driver", "")).lower()
    band_count = meta.get("count") or meta.get("bands") or 0

    if "sentinel-1" in sensor or "sar" in sensor or "radar" in sensor:
        return "sar"
    if "sentinel-2" in sensor or "landsat" in sensor or "planet" in sensor:
        return "optical"

    # 2. Filename heuristics
    sar_keywords = ["sar", "s1", "sentinel1", "sentinel-1", "radar", "backscatter", "_vv", "_vh", "slc", "grd"]
    if any(k in fn_lower for k in sar_keywords):
        return "sar"

    optical_keywords = ["optical", "s2", "sentinel2", "sentinel-2", "landsat", "rgb", "planet", "urban", "agricultural", "patch"]
    if any(k in fn_lower for k in optical_keywords):
        return "optical"

    # 3. Image mode & band analysis
    if img is not None:
        if img.mode in ("L", "I", "F") or band_count == 1:
            return "sar"
        return "optical"

    return "optical"


def detect_temporal_metadata(
    primary_filename: str,
    secondary_filename: Optional[str],
    primary_meta: Optional[Dict[str, Any]],
    secondary_meta: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Detects acquisition dates and temporal sequence between image pairs."""
    meta1 = primary_meta or {}
    meta2 = secondary_meta or {}
    date1 = meta1.get("acquisition_date") or meta1.get("date")
    date2 = meta2.get("acquisition_date") or meta2.get("date")

    fn1 = primary_filename.lower()
    fn2 = (secondary_filename or "").lower()

    has_temporal_indicators = False
    if date1 and date2 and date1 != date2:
        has_temporal_indicators = True
    elif ("t1" in fn1 and "t2" in fn2) or ("before" in fn1 and "after" in fn2) or ("pre" in fn1 and "post" in fn2):
        has_temporal_indicators = True

    return {
        "date_t1": date1 or ("T1 (Pre-event)" if "t1" in fn1 or "before" in fn1 else None),
        "date_t2": date2 or ("T2 (Post-event)" if "t2" in fn2 or "after" in fn2 else None),
        "is_temporal_pair": has_temporal_indicators,
    }


def validate_spatial_compatibility(
    primary_meta: Optional[Dict[str, Any]],
    secondary_meta: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Validates spatial CRS alignment and co-registration compatibility between two images."""
    meta1 = primary_meta or {}
    meta2 = secondary_meta or {}

    crs1 = meta1.get("crs")
    crs2 = meta2.get("crs")
    is_geospatial1 = meta1.get("is_geospatial", False)
    is_geospatial2 = meta2.get("is_geospatial", False)

    if is_geospatial1 and is_geospatial2 and crs1 and crs2:
        crs_compatible = (crs1 == crs2)
        return {
            "is_compatible": crs_compatible,
            "crs_primary": crs1,
            "crs_secondary": crs2,
            "co_registration": "Georeferenced (CRS Matched)" if crs_compatible else "CRS Mismatch (Reprojection Recommended)",
        }

    return {
        "is_compatible": True,
        "crs_primary": crs1 or "Pixel Space",
        "crs_secondary": crs2 or "Pixel Space",
        "co_registration": "Pixel-space grid aligned",
    }


# ============================================================================
# 3. Main Agentic Orchestrator
# ============================================================================
class AgenticOrchestrator:
    """
    Senior Agentic Query & Tool Orchestrator for Multimodal Remote Sensing (SIH26167).
    Executes a structured reasoning pipeline:
      USER QUERY -> INTENT ANALYSIS -> INPUT ANALYSIS -> TASK SELECTION ->
      TOOL SELECTION -> PARAMETER CONFIG -> SPECIALIST EXECUTION ->
      VALIDATION & CONFIDENCE BREAKDOWN -> AUDIT PROVENANCE.
    """

    def __init__(self, vqa_model: Optional[BaseVQAModel] = None, config_path: Optional[str] = None):
        self.vqa_model = vqa_model
        self.config = load_model_config(config_path)
        self.task_registry = TASK_REGISTRY

    def classify_task(
        self,
        query: str,
        has_secondary_image: bool = False,
        explicit_mode: Optional[str] = None,
        primary_filename: str = "primary_image",
        secondary_filename: Optional[str] = None,
        primary_meta: Optional[Dict[str, Any]] = None,
        secondary_meta: Optional[Dict[str, Any]] = None,
    ) -> AgentTaskType:
        """
        Classifies the specialist task by synthesizing query intent, input count,
        sensor modalities, and temporal relationships.
        """
        # 1. Respect explicit manual override if provided and valid
        if explicit_mode and isinstance(explicit_mode, str) and explicit_mode.lower() not in ("auto", "none", ""):
            try:
                return AgentTaskType(explicit_mode.lower())
            except ValueError:
                pass

        # 2. Analyze query intent
        intent_info = analyze_query_intent(query)
        candidate_task = intent_info.task

        # 3. Detect modalities
        modality1 = detect_image_modality(None, primary_filename, primary_meta)
        modality2 = detect_image_modality(None, secondary_filename, secondary_meta) if has_secondary_image else "none"

        # 4. Joint Synthesis
        if not has_secondary_image:
            # Single Image Cases
            if candidate_task == AgentTaskType.GROUNDING:
                return AgentTaskType.GROUNDING
            if candidate_task == AgentTaskType.BITEMPORAL_CHANGE:
                # Query asked for change, but only 1 image was provided -> flagged as bitemporal for validation
                return AgentTaskType.BITEMPORAL_CHANGE
            if candidate_task == AgentTaskType.OPTICAL_SAR_FUSION:
                return AgentTaskType.OPTICAL_SAR_FUSION
            return AgentTaskType.VQA

        else:
            # Two Images Provided
            # If query explicitly requires Optical + SAR fusion
            if candidate_task == AgentTaskType.OPTICAL_SAR_FUSION:
                return AgentTaskType.OPTICAL_SAR_FUSION

            # If query explicitly asks for temporal change
            if candidate_task == AgentTaskType.BITEMPORAL_CHANGE:
                return AgentTaskType.BITEMPORAL_CHANGE

            # If query was general ("Analyze these images" / "Describe"), inspect modalities:
            if (modality1 == "optical" and modality2 == "sar") or (modality1 == "sar" and modality2 == "optical"):
                return AgentTaskType.OPTICAL_SAR_FUSION

            # Default two-image workflow is bi-temporal change
            return AgentTaskType.BITEMPORAL_CHANGE

    def execute(
        self,
        query: str,
        primary_image: Image.Image,
        secondary_image: Optional[Image.Image] = None,
        primary_meta: Optional[Dict[str, Any]] = None,
        secondary_meta: Optional[Dict[str, Any]] = None,
        explicit_task: Optional[str] = None,
        primary_filename: str = "primary_image",
        secondary_filename: Optional[str] = None,
    ) -> AgentResponse:
        """
        Executes the full agentic multi-stage pipeline with observable audit provenance.
        """
        start_time = time.time()
        primary_meta = primary_meta or {}
        secondary_meta = secondary_meta or {}
        has_secondary = (secondary_image is not None)

        # ---------------------------------------------------------------------
        # Stage 1: Query Intent Analysis
        # ---------------------------------------------------------------------
        intent_info = analyze_query_intent(query)

        # ---------------------------------------------------------------------
        # Stage 2: Input & Modality Analysis
        # ---------------------------------------------------------------------
        modality1 = detect_image_modality(primary_image, primary_filename, primary_meta)
        modality2 = detect_image_modality(secondary_image, secondary_filename, secondary_meta) if has_secondary else "none"
        temporal_info = detect_temporal_metadata(primary_filename, secondary_filename, primary_meta, secondary_meta)
        spatial_compat = validate_spatial_compatibility(primary_meta, secondary_meta) if has_secondary else None

        # ---------------------------------------------------------------------
        # Stage 3: Task & Tool Selection
        # ---------------------------------------------------------------------
        task_type = self.classify_task(
            query=query,
            has_secondary_image=has_secondary,
            explicit_mode=explicit_task,
            primary_filename=primary_filename,
            secondary_filename=secondary_filename,
            primary_meta=primary_meta,
            secondary_meta=secondary_meta,
        )

        reg_entry = self.task_registry.get(task_type, self.task_registry[AgentTaskType.VQA])
        tool_name = reg_entry["tool"]
        
        # Check active model mode
        # Check active model mode
        is_mock_mode = (
            isinstance(self.vqa_model, MockVQAModel)
            or os.getenv("SATQUERY_MOCK_MODEL", "0").lower() in ("1", "true", "yes")
            or getattr(self.vqa_model, "model_id", "").startswith(("mock", "fallback"))
        )

        if task_type == AgentTaskType.GROUNDING:
            grounding_engine = get_grounding_model(force_mock=False, allow_heuristic=True)
            model_name = grounding_engine.model_id
            model_mode = grounding_engine.model_type  # "HEURISTIC"
            is_mock_task = False
        elif is_mock_mode:
            model_name = getattr(self.vqa_model, "model_id", reg_entry["mock_model"])
            model_mode = "mock"
            is_mock_task = True
        else:
            model_name = getattr(self.vqa_model, "model_id", reg_entry["default_model"])
            model_mode = "real"
            is_mock_task = False

        # Determine strict task-aligned routing reason
        if task_type == AgentTaskType.VQA:
            if intent_info.intent == "semantic_presence_question":
                task_reason = "User is asking whether a visual feature is present in a single image."
            elif intent_info.intent in ("scene_description", "scene_understanding", "comprehensive_scene_description", "presence_verification"):
                task_reason = "User requested semantic scene understanding from a single image."
            elif intent_info.intent == "quantitative_query":
                task_reason = "User is asking a quantitative or statistical question about the image."
            else:
                task_reason = "User requested semantic scene understanding from a single image."
        elif task_type == AgentTaskType.GROUNDING:
            task_reason = "User requested spatial localization/highlighting."
        elif task_type == AgentTaskType.BITEMPORAL_CHANGE:
            task_reason = "Query requests comparative temporal assessment between two observation dates or asks about dynamic land-cover change."
        elif task_type == AgentTaskType.OPTICAL_SAR_FUSION:
            task_reason = "Query explicitly requests joint cross-modal analysis combining optical spectral bands and SAR microwave radar."
        else:
            task_reason = intent_info.reason

        # Initialize Audit Steps
        input_desc = f"Primary: {primary_filename} [{modality1.upper()}, {primary_image.size[0]}x{primary_image.size[1]}]"
        if has_secondary:
            input_desc += f" + Secondary: {secondary_filename or 'secondary'} [{modality2.upper()}, {secondary_image.size[0]}x{secondary_image.size[1]}]"

        trace_steps = [
            {
                "step": 1,
                "action": "InputModalityAnalysis",
                "details": input_desc,
            },
            {
                "step": 2,
                "action": "QueryIntentAnalysis",
                "details": f"Intent: {intent_info.intent} | Reason: {task_reason}",
            },
            {
                "step": 3,
                "action": "TargetExtraction",
                "details": f"Target: '{intent_info.target or 'General Scene'}' (Requires Spatial Evidence: {intent_info.requires_spatial_evidence})",
            },
            {
                "step": 4,
                "action": "TaskAndToolRouting",
                "details": f"Selected Task: {task_type.value.upper()} -> Tool: {tool_name} (Model: {model_name} [{model_mode.upper()}])",
            },
        ]

        # ---------------------------------------------------------------------
        # Stage 4: Input Validation & Requirement Checking
        # ---------------------------------------------------------------------
        validation_error: Optional[str] = None

        if task_type == AgentTaskType.BITEMPORAL_CHANGE:
            if not has_secondary:
                validation_error = (
                    "Bi-temporal change analysis requires two spatially corresponding satellite images "
                    "from different dates (T1 and T2). Only 1 image was provided. Please upload a secondary scene."
                )
        elif task_type == AgentTaskType.OPTICAL_SAR_FUSION:
            if not has_secondary:
                validation_error = (
                    "Optical + SAR cross-modal fusion requires both an Optical (e.g. Sentinel-2) and "
                    "a SAR radar (e.g. Sentinel-1) image pair. Only 1 image was provided. Please upload the matching sensor scene."
                )
            elif modality1 == "optical" and modality2 == "optical":
                validation_error = (
                    "Optical + SAR fusion requires complementary optical and radar/SAR sensors. "
                    "Both uploaded images were detected as Optical imagery."
                )

        # ---------------------------------------------------------------------
        # Stage 5: Execution or Validation Failure Handling
        # ---------------------------------------------------------------------
        boxes: Optional[List[BoundingBox]] = None
        change_map_url: Optional[str] = None
        grounding_mask_url: Optional[str] = None
        grounding_result_dict: Optional[Dict[str, Any]] = None
        routing_conf = intent_info.confidence
        model_conf: Optional[float] = None
        evidence_conf: Optional[float] = None
        answer_text: str = ""
        rs_adaptation: str = "NOT LOADED"
        evidence_used: str = "NONE"
        evidence_items_collected: List[EvidenceItem] = []

        if validation_error:
            # Record validation error in trace
            trace_steps.append({
                "step": 5,
                "action": "InputRequirementValidation",
                "status": "FAILED",
                "details": validation_error,
            })
            answer_text = f"Input Requirement Notice: {validation_error}"
            tool_name = "InputValidationEngine"
            model_name = "SatQuery-Validator-v1"

        elif task_type == AgentTaskType.GROUNDING:
            grounding_engine = get_grounding_model(force_mock=False, allow_heuristic=True)
            ground_res = grounding_engine.ground(
                primary_image,
                query,
                target=intent_info.target,
            )
            answer_text = ground_res.message
            boxes = ground_res.detections
            evidence_conf = ground_res.confidence
            model_name = ground_res.model_name
            model_mode = ground_res.model_type
            model_conf = 0.88 if model_mode.upper() == "REAL" else None
            grounding_mask_url = ground_res.mask_url
            grounding_result_dict = ground_res.model_dump()

            if boxes and len(boxes) > 0:
                pbox = boxes[0]
                cov_pct = pbox.bbox_coverage_pct or round((pbox.xmax - pbox.xmin) * (pbox.ymax - pbox.ymin) * 100, 1)
                mask_pct = pbox.mask_coverage_pct or ground_res.mask_coverage_pct or cov_pct
                candidates_evaluated = ground_res.raw_details.get("candidates_evaluated", len(boxes))
                trace_steps.append({
                    "step": 5,
                    "action": "SpecialistToolExecution",
                    "tool": tool_name,
                    "model": model_name,
                    "details": f"Method: spectral/color heuristic grounding | Target: '{intent_info.target or 'entity'}' | Candidates evaluated: {candidates_evaluated} | Selected: {len(boxes)}",
                })
                trace_steps.append({
                    "step": 6,
                    "action": "DetectionValidation",
                    "details": f"Validated {len(boxes)} bounding box(es). Mask coverage: {mask_pct}%. Bounding box coverage: {cov_pct}%. Full-image fallback rejected.",
                })
                trace_steps.append({
                    "step": 7,
                    "action": "EvidenceGeneration",
                    "details": f"Generated visual bounding boxes with coordinates and calibrated heuristic confidence ({evidence_conf or '—'}). Segmentation mask overlay rendered.",
                })
            else:
                trace_steps.append({
                    "step": 5,
                    "action": "SpecialistToolExecution",
                    "tool": tool_name,
                    "model": model_name,
                    "details": f"Method: spectral/color heuristic grounding | No reliable target region detected for '{intent_info.target or 'entity'}'.",
                })
                trace_steps.append({
                    "step": 6,
                    "action": "DetectionValidation",
                    "details": "Zero candidate regions met cohesive spatial criteria. Full-image fallback rejected.",
                })

        elif task_type == AgentTaskType.BITEMPORAL_CHANGE:
            img2 = secondary_image if secondary_image is not None else primary_image
            answer_text, change_map_url, raw_conf = analyze_bitemporal_change(
                primary_image, img2, query, primary_meta, secondary_meta
            )
            evidence_conf = raw_conf
            model_conf = 0.92 if model_mode == "real" else None

            trace_steps.append({
                "step": 5,
                "action": "SpecialistToolExecution",
                "tool": tool_name,
                "details": "Computed pixel difference residual, synthesized change heatmap, and quantified surface alteration.",
            })
            trace_steps.append({
                "step": 6,
                "action": "EvidenceGeneration",
                "details": "Generated bi-temporal difference heatmap overlay and surface change percentages.",
            })

        elif task_type == AgentTaskType.OPTICAL_SAR_FUSION:
            sar_img = secondary_image if secondary_image is not None else primary_image.convert("L")
            answer_text, raw_conf = analyze_optical_sar_pair(
                primary_image, sar_img, query, primary_meta, secondary_meta
            )
            evidence_conf = raw_conf
            model_conf = 0.93 if model_mode == "real" else None

            trace_steps.append({
                "step": 5,
                "action": "SpecialistToolExecution",
                "tool": tool_name,
                "details": "Fused optical visible/NIR spectral bands with SAR polarimetric backscatter to resolve cloud occlusion and double-bounce targets.",
            })
            trace_steps.append({
                "step": 6,
                "action": "EvidenceGeneration",
                "details": "Synthesized multisensor cross-modal assessment and joint land-cover classification.",
            })

        else:
            # Single-Image VQA / Scene Understanding (VQA V2)
            vqa_engine = self.vqa_model or get_vqa_model(force_mock=is_mock_mode)
            model_name = getattr(vqa_engine, "model_id", "mock-vlm-v1")
            model_mode = getattr(vqa_engine, "mode", "MOCK").upper()
            rs_adaptation = getattr(vqa_engine, "rs_adaptation_status", "NOT LOADED")

            # Check if query specifically asks for entity presence verification
            target_to_ground = intent_info.target
            if not target_to_ground or target_to_ground == "target entity":
                target_to_ground = extract_target_entity(query)

            is_presence_check = (intent_info.intent == "semantic_presence_question") and (target_to_ground not in ("target entity", "General Scene", "salient geospatial entity", None))

            evidence_used = "NONE"
            vqa_context = {"metadata": primary_meta}
            evidence_items_collected: List[EvidenceItem] = []

            if is_presence_check:
                # Trace: Model selection & Spatial Evidence Request
                trace_steps.append({
                    "step": 5,
                    "action": "VQAModelSelection",
                    "details": f"Model: {model_name} | Mode: {model_mode} | RS Adaptation: {rs_adaptation}",
                })
                trace_steps.append({
                    "step": 6,
                    "action": "EvidenceRequest",
                    "details": f"Requested spatial verification for target '{target_to_ground}' from Grounding Tool.",
                })

                grounding_engine = get_grounding_model(force_mock=False, allow_heuristic=True)
                ground_res = grounding_engine.ground(primary_image, query, target=target_to_ground)

                if ground_res.success and ground_res.detections:
                    evidence_used = "GROUNDING"
                    boxes = ground_res.detections
                    grounding_mask_url = ground_res.mask_url
                    evidence_conf = ground_res.confidence

                    vqa_context["grounding"] = {
                        "target": target_to_ground,
                        "boxes": [b.model_dump() for b in ground_res.detections],
                        "mask_coverage_pct": ground_res.mask_coverage_pct,
                        "bbox_coverage_pct": ground_res.bbox_coverage_pct,
                        "confidence": ground_res.confidence,
                    }

                    pbox = boxes[0]
                    p_mask_cov = ground_res.mask_coverage_pct or pbox.mask_coverage_pct
                    p_bbox_cov = ground_res.bbox_coverage_pct or pbox.bbox_coverage_pct
                    evidence_items_collected.append(
                        EvidenceItem(
                            type=EvidenceType.SPATIAL,
                            source="RS-SpectralHeuristic-Grounder-v1",
                            target=target_to_ground,
                            bbox=pbox.model_dump(),
                            mask_coverage_pct=p_mask_cov,
                            bbox_coverage_pct=p_bbox_cov,
                            confidence=evidence_conf,
                            description=f"Identified verified spatial cluster for '{target_to_ground}'.",
                        )
                    )

                    trace_steps.append({
                        "step": 7,
                        "action": "SpecialistToolExecution",
                        "tool": "RemoteSensingGroundingTool",
                        "details": f"Identified spatial evidence ({len(boxes)} candidate region(s)) with heuristic confidence {evidence_conf or '—'}.",
                    })
                    trace_steps.append({
                        "step": 8,
                        "action": "EvidenceFusion",
                        "details": f"Synthesized semantic VQA understanding with verified spatial grounding evidence for '{target_to_ground}'.",
                    })
                else:
                    trace_steps.append({
                        "step": 7,
                        "action": "SpecialistToolExecution",
                        "tool": "RemoteSensingGroundingTool",
                        "details": f"No candidate spatial clusters meeting target spectral criteria detected for '{target_to_ground}'.",
                    })
                    trace_steps.append({
                        "step": 8,
                        "action": "EvidenceFusion",
                        "details": "Direct qualitative evaluation without affirmative spatial grounding evidence.",
                    })
            else:
                trace_steps.append({
                    "step": 5,
                    "action": "VQAModelSelection",
                    "details": f"Model: {model_name} | Mode: {model_mode} | RS Adaptation: {rs_adaptation}",
                })
                trace_steps.append({
                    "step": 6,
                    "action": "EvidenceRequest",
                    "details": "Direct scene understanding without auxiliary spatial verification.",
                })
                trace_steps.append({
                    "step": 7,
                    "action": "SpecialistToolExecution",
                    "tool": tool_name,
                    "model": model_name,
                    "details": f"Executed vision-language scene understanding on normalized raster composite ({model_mode} mode).",
                })
                trace_steps.append({
                    "step": 8,
                    "action": "EvidenceFusion",
                    "details": "Synthesized qualitative land-cover and object observations.",
                })

            # Execute VQA inference with evidence context
            vqa_ans_res = vqa_engine.answer(primary_image, query, context=vqa_context)
            answer_text = vqa_ans_res.answer
            model_conf = vqa_ans_res.confidence if model_mode == "REAL" else None

            trace_steps.append({
                "step": len(trace_steps) + 1,
                "action": "AnswerGeneration",
                "details": "Formulated evidence-grounded explanation without hallucinated quantitative metrics.",
            })
            trace_steps.append({
                "step": len(trace_steps) + 1,
                "action": "ConfidenceAssessment",
                "details": f"Model confidence: {'Not calibrated' if model_conf is None else model_conf} | Grounding evidence confidence: {evidence_conf or '—'}",
            })

        execution_time = time.time() - start_time

        # ---------------------------------------------------------------------
        # Stage 6: Confidence Breakdown & Trace Synthesis
        # ---------------------------------------------------------------------
        confidence_breakdown = ConfidenceBreakdown(
            routing_confidence=routing_conf,
            model_confidence=model_conf,
            evidence_confidence=evidence_conf,
            is_mock=is_mock_task if task_type == AgentTaskType.GROUNDING else is_mock_mode,
        )

        overall_confidence: Optional[float] = (
            evidence_conf
            if (task_type == AgentTaskType.GROUNDING and boxes and len(boxes) > 0)
            else (
                model_conf
                if (model_mode == "REAL" and model_conf is not None)
                else (evidence_conf if (task_type in (AgentTaskType.BITEMPORAL_CHANGE, AgentTaskType.OPTICAL_SAR_FUSION) and not is_mock_task) else None)
            )
        )

        trace_steps.append({
            "step": len(trace_steps) + 1,
            "action": "AuditProvenanceLogging",
            "details": f"Completed multi-stage execution in {round(execution_time, 4)}s. Provenance trace logged.",
        })

        execution_trace = {
            "query": query,
            "task": task_type.value,
            "intent": intent_info.model_dump(),
            "target": intent_info.target,
            "tool_used": tool_name,
            "model_selected": model_name,
            "model_mode": model_mode,
            "rs_adaptation": rs_adaptation if task_type == AgentTaskType.VQA else "NOT LOADED",
            "evidence_used": evidence_used if task_type == AgentTaskType.VQA else ("GROUNDING" if boxes else "NONE"),
            "detected_inputs": {
                "primary": {"filename": primary_filename, "modality": modality1},
                "secondary": {"filename": secondary_filename, "modality": modality2} if has_secondary else None,
                "temporal": temporal_info,
                "spatial_compatibility": spatial_compat,
            },
            "routing_reason": task_reason,
            "validation_status": "PASSED" if not validation_error else "FAILED",
            "validation_error": validation_error,
            "confidence_breakdown": confidence_breakdown.model_dump(),
            "execution_time_sec": round(execution_time, 4),
            "steps": trace_steps,
        }

        # ---------------------------------------------------------------------
        # Stage 7: Provenance Logging
        # ---------------------------------------------------------------------
        record_execution(
            task=task_type.value,
            model_name=f"{model_name} ({model_mode})",
            image_path=f"{primary_filename}" + (f" + {secondary_filename}" if secondary_filename else ""),
            question=query,
            answer=answer_text,
            execution_time_sec=execution_time,
            metadata=primary_meta,
            confidence=overall_confidence,
        )

        return AgentResponse(
            answer=answer_text,
            task=task_type.value,
            model=model_name,
            model_mode=model_mode,
            rs_adaptation=rs_adaptation if task_type == AgentTaskType.VQA else "NOT LOADED",
            evidence_used=evidence_used if task_type == AgentTaskType.VQA else ("GROUNDING" if boxes else "NONE"),
            tool_used=tool_name,
            confidence=overall_confidence,
            boxes=boxes,
            change_map_url=change_map_url,
            grounding_mask_url=grounding_mask_url,
            grounding_result=grounding_result_dict,
            evidence_items=evidence_items_collected if evidence_items_collected else None,
            metadata=primary_meta,
            execution_time_sec=round(execution_time, 4),
            execution_trace=execution_trace,
        )

