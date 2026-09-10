from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class AgentTaskType(str, Enum):
    VQA = "vqa"
    GROUNDING = "grounding"
    BITEMPORAL_CHANGE = "bitemporal_change"
    OPTICAL_SAR_FUSION = "optical_sar_fusion"
    VALIDATION_ERROR = "validation_error"


class ConfidenceBreakdown(BaseModel):
    """Calibrated confidence breakdown separating routing, model, and evidence."""
    routing_confidence: float = Field(..., description="Confidence in task intent routing")
    model_confidence: Optional[float] = Field(None, description="Confidence reported by active model, or None if uncalibrated")
    evidence_confidence: Optional[float] = Field(None, description="Confidence based on extracted spatial/spectral evidence")
    is_mock: bool = Field(False, description="True if executed via a mock or prototype engine")


class IntentAnalysis(BaseModel):
    """Structured query intent analysis."""
    task: AgentTaskType = Field(..., description="Selected specialist task")
    intent: str = Field(..., description="Fine-grained semantic intent")
    target: Optional[str] = Field(None, description="Extracted target entity (e.g., water body, buildings)")
    requires_spatial_evidence: bool = Field(False, description="Whether spatial bounding or mask is required")
    requires_multiple_images: bool = Field(False, description="Whether multi-image input is mandatory")
    confidence: float = Field(..., description="Calibrated intent classification confidence")
    reason: str = Field(..., description="Human-auditable explanation for the routing decision")


class BoundingBox(BaseModel):
    """Normalized spatial bounding box coordinates [0.0 - 1.0] for visual grounding."""
    ymin: float = Field(..., description="Top coordinate (0.0 to 1.0)")
    xmin: float = Field(..., description="Left coordinate (0.0 to 1.0)")
    ymax: float = Field(..., description="Bottom coordinate (0.0 to 1.0)")
    xmax: float = Field(..., description="Right coordinate (0.0 to 1.0)")
    label: Optional[str] = Field("target", description="Class or description of grounded entity")
    confidence: Optional[float] = Field(None, description="Confidence score for this detection")
    bbox_coverage_pct: Optional[float] = Field(None, description="Bounding box coverage percentage")
    mask_coverage_pct: Optional[float] = Field(None, description="Detected segmentation mask coverage percentage")
    confidence_type: Optional[str] = Field("heuristic", description="Confidence scoring methodology")


class GeoMetadata(BaseModel):
    """Geospatial metadata extracted from input image."""
    crs: Optional[str] = Field(None, description="Coordinate Reference System e.g. EPSG:32633")
    transform: Optional[list] = Field(None, description="Affine Geotransform matrix")
    bounds: Optional[Dict[str, float]] = Field(None, description="Bounding coordinates (left, bottom, right, top)")
    shape: Optional[list] = Field(None, description="Image dimensions [bands, height, width]")
    count: Optional[int] = Field(None, description="Number of raster bands")
    driver: Optional[str] = Field(None, description="Image driver format (GTiff, PNG, JPEG)")
    dtypes: Optional[list] = Field(None, description="Data type of each band")
    nodata: Optional[Any] = Field(None, description="NoData pixel value")
    is_geospatial: bool = Field(False, description="True if georeferenced metadata is present")


class VQARequest(BaseModel):
    """Input payload for Visual Question Answering."""
    question: str = Field(..., description="Natural language question about the satellite image")
    image_path: Optional[str] = Field(None, description="Local path to satellite image file")


class VQAResponse(BaseModel):
    """Output response from VQA inference."""
    answer: str = Field(..., description="Model generated natural language answer")
    model: str = Field(..., description="Name and version of the vision-language model used")
    model_mode: Optional[str] = Field("MOCK", description="Model mode: REAL, MOCK, ADAPTED, or HEURISTIC")
    rs_adaptation: Optional[str] = Field("NOT LOADED", description="Remote sensing adaptation status")
    confidence: Optional[float] = Field(None, description="Calibrated confidence score if supported by model, null otherwise")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Image and execution metadata")
    execution_time_sec: Optional[float] = Field(None, description="Total inference time in seconds")
    boxes: Optional[List[BoundingBox]] = Field(default=None, description="Visual grounding bounding boxes if applicable")
    task_type: Optional[str] = Field("vqa", description="Identified remote sensing task type")


from satquery.schemas.evidence import EvidenceItem, EvidenceType



class VQAModelInfo(BaseModel):
    """Detailed model metadata and adaptation provenance."""
    name: str = Field(..., description="Active model identifier")
    mode: str = Field(..., description="Execution mode: REAL, MOCK, ADAPTED, or HEURISTIC")
    remote_sensing_adapted: bool = Field(False, description="True only if adapted RS checkpoint/LoRA is active")
    rs_adaptation_status: str = Field("NOT LOADED", description="Status label for domain adaptation (LOADED / NOT LOADED)")


class VQAAnswerResult(BaseModel):
    """Internal result schema for VQA model inference."""
    answer: str = Field(..., description="Generated natural language explanation")
    model_info: VQAModelInfo = Field(..., description="Model provenance and adaptation info")
    confidence: Optional[float] = Field(None, description="Calibrated model confidence score if supported, null otherwise")
    evidence: List[EvidenceItem] = Field(default_factory=list, description="Verified evidence items utilized in answer generation")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Execution and image metadata")

    @property
    def model_mode(self) -> str:
        return self.model_info.mode

    @property
    def rs_adaptation(self) -> str:
        return self.model_info.rs_adaptation_status


class AgentResponse(BaseModel):
    """Unified response from the Agentic Orchestrator."""
    answer: str = Field(..., description="Evidence-grounded natural language explanation")
    task: str = Field(..., description="Selected specialist task (vqa, grounding, bitemporal_change, optical_sar_fusion)")
    model: str = Field(..., description="Specialist model or fusion engine executed")
    model_mode: Optional[str] = Field(None, description="Model mode: REAL, MOCK, ADAPTED, or HEURISTIC")
    rs_adaptation: Optional[str] = Field(None, description="Remote sensing adaptation status (LOADED / NOT LOADED)")
    evidence_used: Optional[str] = Field(None, description="Primary evidence source (GROUNDING, SEGMENTATION, NONE)")
    tool_used: str = Field(..., description="Tool name invoked by the agent")
    confidence: Optional[float] = Field(None, description="Estimated confidence score")
    confidence_level: Optional[str] = Field(default=None, description="Qualitative confidence label (High, Moderate, Low, Uncalibrated)")
    headline: Optional[str] = Field(default=None, description="Short user-friendly headline summary")
    details: Optional[str] = Field(default=None, description="Simple non-technical breakdown")
    location_summary: Optional[str] = Field(default=None, description="Natural-language location description")
    visual_summary: Optional[str] = Field(default=None, description="Description of visual highlight overlay")
    boxes: Optional[List[BoundingBox]] = Field(default=None, description="Visual grounding bounding boxes")
    change_map_url: Optional[str] = Field(default=None, description="Base64 PNG or URL of change heatmap overlay")
    grounding_mask_url: Optional[str] = Field(default=None, description="Base64 PNG of segmentation mask overlay")
    grounding_result: Optional[Dict[str, Any]] = Field(default=None, description="Full structured GroundingResult payload")
    evidence_items: Optional[List[EvidenceItem]] = Field(default=None, description="List of verified evidence items used")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Primary and secondary image metadata")
    execution_time_sec: float = Field(..., description="Total execution latency in seconds")
    execution_trace: Dict[str, Any] = Field(default_factory=dict, description="Observable step-by-step agentic execution trace")


class ExecutionRecord(BaseModel):
    """Provenance trace schema for recording execution logs."""
    task: str
    model: str
    input: str
    question: str
    output: str
    confidence: Optional[float] = None
    execution_time_sec: float
    timestamp: str
    metadata: Dict[str, Any]
    boxes: Optional[List[Dict[str, Any]]] = None
