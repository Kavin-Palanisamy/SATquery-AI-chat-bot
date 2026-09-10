from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from satquery.schemas.models import ModelProvenance


class GroundingBox(BaseModel):
    """Normalized spatial bounding box [ymin, xmin, ymax, xmax] in range [0.0, 1.0]."""
    ymin: float = Field(..., description="Top boundary (0.0 to 1.0)")
    xmin: float = Field(..., description="Left boundary (0.0 to 1.0)")
    ymax: float = Field(..., description="Bottom boundary (0.0 to 1.0)")
    xmax: float = Field(..., description="Right boundary (0.0 to 1.0)")
    label: str = Field(..., description="Target category or entity name")
    confidence: Optional[float] = Field(None, description="Confidence score for this candidate")
    bbox_coverage_pct: float = Field(..., description="Bounding box coverage percentage relative to entire scene footprint")
    mask_coverage_pct: Optional[float] = Field(None, description="Precise pixel segmentation coverage percentage within the region")
    pixel_coords: Optional[Dict[str, int]] = Field(None, description="Absolute pixel bounding box [xmin, ymin, xmax, ymax]")
    confidence_type: str = Field("heuristic", description="Confidence scoring origin (heuristic / calibrated)")


class GroundingResult(BaseModel):
    """Full structured output from GroundingEngine V3."""
    success: bool = Field(..., description="True if candidate regions were identified")
    target: str = Field(..., description="Entity or feature queried for spatial grounding")
    detections: List[GroundingBox] = Field(default_factory=list, description="List of validated spatial bounding boxes")
    pixel_area: Optional[int] = Field(None, description="Total positive mask pixel count across all detections")
    bbox_area_pixels: Optional[int] = Field(None, description="Total area enclosed by bounding boxes in pixels")
    mask_coverage_pct: Optional[float] = Field(None, description="Strict pixel mask coverage percentage of valid image area")
    bbox_coverage_pct: Optional[float] = Field(None, description="Bounding box coverage percentage of image footprint")
    mask_url: Optional[str] = Field(None, description="Base64 encoded PNG overlay of the binary segmentation mask")
    method: str = Field(..., description="Grounding method executed (e.g., 'spectral_heuristic', 'vlm_grounding')")
    confidence: Optional[float] = Field(None, description="Overall grounding confidence score")
    provenance: ModelProvenance = Field(ModelProvenance.HEURISTIC, description="Grounding model provenance")
    evidence_type: str = Field("BBOX", description="Primary evidence modality emitted (BBOX, MASK)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Image dimensions and execution telemetry")
