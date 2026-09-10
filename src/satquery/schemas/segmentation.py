from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from satquery.schemas.models import ModelProvenance


class LandCoverClass(str, Enum):
    WATER = "water"
    VEGETATION = "vegetation"
    BUILT_UP = "built_up"
    BARE_SOIL = "bare_soil"
    ROADS = "roads"
    AGRICULTURE = "agriculture"
    FOREST = "forest"
    GENERIC = "generic"


class SegmentationResult(BaseModel):
    """Structured output from SegmentationEngine V3."""
    success: bool = Field(..., description="True if segmentation mask was generated")
    target_class: str = Field(..., description="Target land-cover or feature class segmented")
    positive_pixels: int = Field(..., description="Count of positive target pixels")
    valid_pixels: int = Field(..., description="Count of valid non-nodata pixels in image footprint")
    area_percent: float = Field(..., description="Exact measured percentage of scene covered by target class: (positive_pixels / valid_pixels) * 100")
    mask_url: Optional[str] = Field(None, description="Base64 encoded PNG of the binary or multi-class mask overlay")
    method: str = Field(..., description="Segmentation method executed (e.g., 'spectral_ndwi_heuristic', 'sam_geospatial', 'unet')")
    confidence: Optional[float] = Field(None, description="Confidence score for this segmentation")
    provenance: ModelProvenance = Field(ModelProvenance.HEURISTIC, description="Segmentation engine provenance")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Execution and thresholding metadata")
