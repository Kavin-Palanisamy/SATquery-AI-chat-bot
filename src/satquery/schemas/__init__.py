"""
SatQuery AI Unified Common Data Model (Phase 2).
"""
from satquery.schemas.confidence import CalibrationStatus, ConfidenceRecord
from satquery.schemas.evidence import EvidenceItem, EvidenceType
from satquery.schemas.models import ModelInfo, ModelProvenance, ModelStatus
from satquery.schemas.grounding import GroundingBox, GroundingResult
from satquery.schemas.segmentation import LandCoverClass, SegmentationResult
from satquery.schemas.detection import DetectedObject, DetectionResult
from satquery.schemas.change import ChangeRegion, ChangeResult, ChangeType
from satquery.schemas.fusion import FusedRegion, OpticalSARFusionResult
from satquery.schemas.intent import AgentTaskType, QueryIntent
from satquery.schemas.execution import ExecutionStep, ExecutionTrace
from satquery.schemas.validation import InputMetadata, ValidationResult
from satquery.schemas.plan import ExecutionPlan, ToolInfo
from satquery.schemas.caption import CaptionResult
from satquery.schemas.vqa import (
    AgentResponse,
    BoundingBox,
    ConfidenceBreakdown,
    ExecutionRecord,
    GeoMetadata,
    IntentAnalysis,
    VQAAnswerResult,
    VQAModelInfo,
    VQARequest,
    VQAResponse,
)

# Phase 2 Semantic Aliases
VQAResult = VQAAnswerResult
OpticalSARResult = OpticalSARFusionResult
Evidence = EvidenceItem
FinalResponse = AgentResponse

__all__ = [
    "CalibrationStatus",
    "ConfidenceRecord",
    "Evidence",
    "EvidenceItem",
    "EvidenceType",
    "ModelInfo",
    "ModelProvenance",
    "ModelStatus",
    "ToolInfo",
    "ExecutionPlan",
    "InputMetadata",
    "ValidationResult",
    "GroundingBox",
    "GroundingResult",
    "LandCoverClass",
    "SegmentationResult",
    "DetectedObject",
    "DetectionResult",
    "ChangeRegion",
    "ChangeResult",
    "ChangeType",
    "FusedRegion",
    "OpticalSARResult",
    "OpticalSARFusionResult",
    "AgentTaskType",
    "QueryIntent",
    "ExecutionStep",
    "ExecutionTrace",
    "AgentResponse",
    "FinalResponse",
    "CaptionResult",
    "BoundingBox",
    "ConfidenceBreakdown",
    "ExecutionRecord",
    "GeoMetadata",
    "IntentAnalysis",
    "VQAResult",
    "VQAAnswerResult",
    "VQAModelInfo",
    "VQARequest",
    "VQAResponse",
]
