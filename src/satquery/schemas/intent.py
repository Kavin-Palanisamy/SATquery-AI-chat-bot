from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class AgentTaskType(str, Enum):
    VQA = "vqa"
    SCENE_DESCRIPTION = "scene_description"
    GROUNDING = "grounding"
    SEGMENTATION_AREA = "segmentation_area"
    DETECTION_COUNT = "detection_count"
    BITEMPORAL_CHANGE = "bitemporal_change"
    OPTICAL_SAR_FUSION = "optical_sar_fusion"
    INPUT_VALIDATION = "input_validation"
    VALIDATION_ERROR = "validation_error"


class QueryIntent(BaseModel):
    """Fine-grained multi-modal query intent representation."""
    task: AgentTaskType = Field(..., description="Primary specialist task")
    subtask: Optional[str] = Field(None, description="Specific subtask (e.g., 'built_up_expansion', 'water_area', 'building_count')")
    intent: str = Field(..., description="Semantic intent classification string")
    target: Optional[str] = Field(None, description="Extracted target entity or land-cover class")
    modality_requirement: List[str] = Field(default_factory=lambda: ["OPTICAL"], description="Required input modalities (OPTICAL, SAR, MULTISPECTRAL)")
    temporal_requirement: bool = Field(False, description="True if multi-temporal image pair is required")
    spatial_requirement: bool = Field(False, description="True if spatial bounding box or mask overlay is required")
    quantitative_requirement: bool = Field(False, description="True if numeric measurement / counting is requested")
    evidence_requirement: str = Field("NONE", description="Expected evidence source: NONE, GROUNDING, SEGMENTATION, DETECTION, CHANGE, FUSION")
    confidence: float = Field(0.95, description="Intent classification confidence")
    reason: str = Field(..., description="Human-auditable rationale for intent classification")
