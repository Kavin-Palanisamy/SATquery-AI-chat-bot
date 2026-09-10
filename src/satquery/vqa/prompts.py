"""
SatQuery AI — Remote Sensing VQA Prompting Templates & Context Builder.
Problem Statement: SIH26167
"""

from typing import Any, Dict, List, Optional

SYSTEM_RS_VQA_PROMPT = """You are a specialized Remote Sensing and Earth Observation multimodal AI assistant.
Analyze the supplied satellite image carefully.

Core Guidelines:
1. Use accurate remote-sensing and geospatial terminology (e.g., spectral reflectance, NIR signature, urban footprint, canopy vigor, morphological texture).
2. Distinguish direct visual observation from measured evidence.
3. DO NOT fabricate or guess exact object counts, percentage area coverages, geographic coordinates, bounding boxes, or spatial measurements unless they are explicitly provided in the verified evidence context.
4. If the question asks for a quantitative metric (such as area percentage or exact object count) and no verified segmentation/measurement evidence is provided, explicitly state that a dedicated segmentation or detection model is required.
5. If the image lacks sufficient resolution or spectral clarity to confirm a feature, state the limitation honestly.
"""


def format_vqa_prompt(
    question: str,
    context: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Constructs an evidence-aware prompt for the Remote-Sensing VLM.
    """
    clean_q = question.strip()
    if not context:
        return f"Question: {clean_q}\nAnswer concisely and factually based only on observable visual features in the satellite image:"

    evidence_lines: List[str] = []
    
    # 1. Spatial Grounding Evidence
    grounding_data = context.get("grounding")
    if grounding_data and isinstance(grounding_data, dict):
        tgt = grounding_data.get("target", "feature")
        boxes = grounding_data.get("boxes", [])
        mask_cov = grounding_data.get("mask_coverage_pct")
        bbox_cov = grounding_data.get("bbox_coverage_pct")
        conf = grounding_data.get("confidence")
        if boxes:
            pbox = boxes[0]
            loc_desc = f"span [{int(pbox.get('xmin', 0)*100)}%-{int(pbox.get('xmax', 1)*100)}% X, {int(pbox.get('ymin', 0)*100)}%-{int(pbox.get('ymax', 1)*100)}% Y]"
            cov_desc = f", mask coverage: {mask_cov}%" if mask_cov is not None else (f", bbox coverage: {bbox_cov}%" if bbox_cov is not None else "")
            evidence_lines.append(f"- Spatial Grounding Tool: Localized primary '{tgt}' region at {loc_desc}{cov_desc} (confidence: {conf or 'uncalibrated'}).")

    # 2. Segmentation Evidence
    seg_data = context.get("segmentation")
    if seg_data and isinstance(seg_data, dict):
        class_covs = seg_data.get("class_coverages", {})
        cov_str = ", ".join([f"{k}: {v}%" for k, v in class_covs.items()])
        evidence_lines.append(f"- Segmentation Engine: Measured land-cover distribution: {cov_str}.")

    # 3. Metadata
    meta = context.get("metadata")
    if meta and isinstance(meta, dict):
        crs = meta.get("crs")
        sensor = meta.get("sensor")
        if crs:
            evidence_lines.append(f"- Metadata: CRS: {crs}, Sensor: {sensor or 'Earth Observation Optical'}.")

    if not evidence_lines:
        return f"Question: {clean_q}\nAnswer concisely and factually based on observable visual features in the satellite image:"

    evidence_block = "\n".join(evidence_lines)
    return (
        f"VERIFIED EVIDENCE CONTEXT:\n{evidence_block}\n\n"
        f"Question: {clean_q}\n"
        f"Answer concisely, directly addressing the question and citing the provided verified evidence context where applicable:"
    )
