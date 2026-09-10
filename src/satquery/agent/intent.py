import re
from typing import Any, Dict, List, Optional
from satquery.schemas.intent import AgentTaskType, QueryIntent


def parse_query_intent(query: str, num_images: int = 1, modalities: Optional[List[str]] = None) -> QueryIntent:
    """
    Analyzes natural language queries and available inputs to produce structured QueryIntent.
    """
    clean_q = query.strip()
    q_lower = clean_q.lower()
    modalities = [m.upper() for m in (modalities or ["OPTICAL"])]

    # 1. Target Entity Extraction
    target = None
    target_patterns = [
        (r"\b(water(?:\s*bod(?:y|ies))?|lake|river|reservoir|ocean|sea|canal|pond)\b", "water"),
        (r"\b(building|buildings|house|houses|urban|structure|structures|built-up|residential)\b", "buildings"),
        (r"\b(road|roads|highway|street|runway|airport|pathway)\b", "road / runway"),
        (r"\b(vegetation|canopy|forest|crop|crops|agricultural|farmland|trees)\b", "vegetation"),
        (r"\b(ship|ships|vessel|vessels|boat|boats|port|harbor)\b", "vessel / harbor"),
        (r"\b(solar\s*park|solar\s*panels?|photovoltaic)\b", "solar park"),
    ]

    for pattern, label in target_patterns:
        if re.search(pattern, q_lower):
            target = label
            break

    # -------------------------------------------------------------
    # INTENT CLASSIFICATION RULES
    # -------------------------------------------------------------

    # A. Optical + SAR Cross-Modal Fusion
    sar_terms = ["sar", "radar", "microwave", "backscatter", "penetrat", "cross-modal", "both sensors", "optical and sar", "together"]
    if any(term in q_lower for term in sar_terms) or (num_images >= 2 and "SAR" in modalities and "OPTICAL" in modalities):
        return QueryIntent(
            task=AgentTaskType.OPTICAL_SAR_FUSION,
            subtask="crossmodal_synergy",
            intent="crossmodal_multisensor_fusion",
            target=target or "cross_modal_features",
            modality_requirement=["OPTICAL", "SAR"],
            temporal_requirement=False,
            spatial_requirement=True,
            quantitative_requirement=False,
            evidence_requirement="FUSION",
            confidence=0.96,
            reason="Query explicitly requests joint cross-modal analysis combining optical spectral bands and SAR microwave radar.",
        )

    # B. Bi-Temporal Change Detection
    change_terms = ["change", "changed", "difference", "between these two", "between t1 and t2", "between the two", "increased", "decreased", "growth", "flood", "loss", "expansion", "alteration"]
    if any(term in q_lower for term in change_terms) or num_images >= 2:
        subtask = "generic_change"
        if "built-up" in q_lower or "urban" in q_lower or "building" in q_lower:
            subtask = "built_up_expansion"
        elif "vegetation" in q_lower or "forest" in q_lower:
            subtask = "vegetation_dynamics"
        elif "water" in q_lower or "flood" in q_lower:
            subtask = "hydrologic_inundation"

        return QueryIntent(
            task=AgentTaskType.BITEMPORAL_CHANGE,
            subtask=subtask,
            intent="temporal_change_detection",
            target=target or "land_cover_dynamics",
            modality_requirement=["OPTICAL"],
            temporal_requirement=True,
            spatial_requirement=True,
            quantitative_requirement=True,
            evidence_requirement="CHANGE",
            confidence=0.95,
            reason="Query requests comparative temporal assessment between two observation dates or asks about dynamic land-cover change.",
        )

    # C. Spatial Grounding / Localization / Highlighting
    grounding_verbs = ["find", "locate", "highlight", "draw a box", "draw box", "bound", "where is", "where are", "show me", "detect and show", "pinpoint"]
    if any(re.search(r"\b" + re.escape(v) + r"\b", q_lower) for v in grounding_verbs):
        return QueryIntent(
            task=AgentTaskType.GROUNDING,
            subtask="bounding_box_localization",
            intent="spatial_localization",
            target=target or "target entity",
            modality_requirement=["OPTICAL"],
            temporal_requirement=False,
            spatial_requirement=True,
            quantitative_requirement=False,
            evidence_requirement="GROUNDING",
            confidence=0.94,
            reason="User requested spatial localization/highlighting.",
        )

    # D. Quantitative Area / Percentage Calculation
    if any(w in q_lower for w in ["percentage", "percent", "%", "coverage", "how much of the image", "area fraction"]):
        return QueryIntent(
            task=AgentTaskType.SEGMENTATION_AREA,
            subtask="pixel_area_measurement",
            intent="quantitative_percentage_query",
            target=target or "land_cover_class",
            modality_requirement=["OPTICAL"],
            temporal_requirement=False,
            spatial_requirement=True,
            quantitative_requirement=True,
            evidence_requirement="SEGMENTATION",
            confidence=0.96,
            reason="User is asking a quantitative or statistical coverage question about the image footprint.",
        )

    # E. Discrete Object Counting
    if any(w in q_lower for w in ["how many", "count", "number of"]):
        return QueryIntent(
            task=AgentTaskType.DETECTION_COUNT,
            subtask="instance_counting",
            intent="quantitative_counting_query",
            target=target or "object_instances",
            modality_requirement=["OPTICAL"],
            temporal_requirement=False,
            spatial_requirement=True,
            quantitative_requirement=True,
            evidence_requirement="DETECTION",
            confidence=0.96,
            reason="User is asking for an exact count of discrete object instances in the scene.",
        )

    # F. Holistic Scene Description / Overview
    if any(w in q_lower for w in ["describe", "overview", "land cover", "what is visible", "major objects", "scene understanding"]):
        return QueryIntent(
            task=AgentTaskType.VQA,
            subtask="scene_description",
            intent="scene_description",
            target=target or "scene",
            modality_requirement=["OPTICAL"],
            temporal_requirement=False,
            spatial_requirement=False,
            quantitative_requirement=False,
            evidence_requirement="NONE",
            confidence=0.96,
            reason="User requested semantic scene understanding from a single image.",
        )

    # G. Semantic Feature Presence VQA
    return QueryIntent(
        task=AgentTaskType.VQA,
        subtask="semantic_presence",
        intent="semantic_presence_question",
        target=target or "feature",
        modality_requirement=["OPTICAL"],
        temporal_requirement=False,
        spatial_requirement=False,
        quantitative_requirement=False,
        evidence_requirement="NONE",
        confidence=0.96,
        reason="User is asking whether a visual feature is present in a single image.",
    )
