from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from satquery.schemas.grounding import GroundingBox
from satquery.schemas.models import ModelProvenance


class DetectedObject(BaseModel):
    """Discrete detected object instance."""
    label: str = Field(..., description="Object category (e.g., 'building', 'ship', 'aircraft', 'storage_tank')")
    bbox: GroundingBox = Field(..., description="Spatial bounding box enclosing the object")
    confidence: Optional[float] = Field(None, description="Detector confidence score for this instance")
    area_pixels: Optional[int] = Field(None, description="Footprint area in pixels")


class DetectionResult(BaseModel):
    """Structured output from DetectionEngine V3."""
    success: bool = Field(..., description="True if detector executed successfully")
    target_class: str = Field(..., description="Target object class requested for detection")
    total_count: int = Field(..., description="Exact discrete count of detected object instances")
    objects: List[DetectedObject] = Field(default_factory=list, description="List of detected individual instances")
    method: str = Field(..., description="Detection method executed (e.g., 'contour_component_detector', 'yolo_rs', 'dota_detector')")
    confidence: Optional[float] = Field(None, description="Overall detection confidence score")
    provenance: ModelProvenance = Field(ModelProvenance.HEURISTIC, description="Detection engine provenance")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Image dimensions and counting metadata")
