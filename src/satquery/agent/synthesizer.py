from typing import Any, Dict, List, Optional
from satquery.schemas.confidence import CalibrationStatus, ConfidenceRecord
from satquery.schemas.evidence import EvidenceItem, EvidenceType
from satquery.schemas.vqa import AgentResponse, BoundingBox


class AnswerSynthesizer:
    """
    Synthesizes specialist tool outputs, verified evidence, limitations, and provenance into unified AgentResponse.
    """

    @staticmethod
    def synthesize(
        task: str,
        answer: str,
        tool_used: str,
        model_name: str,
        model_mode: str,
        rs_adaptation: str,
        confidence: Optional[float],
        evidence_used: Optional[str] = "NONE",
        evidence_items: Optional[List[EvidenceItem]] = None,
        boxes: Optional[List[BoundingBox]] = None,
        change_map_url: Optional[str] = None,
        grounding_mask_url: Optional[str] = None,
        grounding_result: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        execution_time_sec: float = 0.0,
        execution_trace: Optional[Dict[str, Any]] = None,
        limitations: Optional[List[str]] = None,
    ) -> AgentResponse:
        evidence_items = evidence_items or []
        metadata = metadata or {}
        execution_trace = execution_trace or {}
        limitations = limitations or []

        if limitations:
            metadata["limitations"] = limitations

        return AgentResponse(
            answer=answer,
            task=task,
            model=model_name,
            model_mode=model_mode,
            rs_adaptation=rs_adaptation,
            evidence_used=evidence_used,
            tool_used=tool_used,
            confidence=confidence,
            boxes=boxes,
            change_map_url=change_map_url,
            grounding_mask_url=grounding_mask_url,
            grounding_result=grounding_result,
            evidence_items=evidence_items,
            metadata=metadata,
            execution_time_sec=round(execution_time_sec, 3),
            execution_trace=execution_trace,
        )
