from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from satquery.schemas.models import ModelProvenance


class ChangeType(str, Enum):
    BUILT_UP_EXPANSION = "built_up_expansion"
    VEGETATION_LOSS = "vegetation_loss"
    VEGETATION_GAIN = "vegetation_gain"
    WATER_EXPANSION = "water_expansion"
    WATER_REDUCTION = "water_reduction"
    GENERIC_SPECTRAL_CHANGE = "generic_spectral_change"
    NO_SIGNIFICANT_CHANGE = "no_significant_change"


class ChangeRegion(BaseModel):
    """Localized geographic region of identified temporal change."""
    region_id: int = Field(..., description="Region index identifier")
    ymin: float = Field(..., description="Top boundary (0.0 to 1.0)")
    xmin: float = Field(..., description="Left boundary (0.0 to 1.0)")
    ymax: float = Field(..., description="Bottom boundary (0.0 to 1.0)")
    xmax: float = Field(..., description="Right boundary (0.0 to 1.0)")
    change_type: ChangeType = Field(..., description="Classified category of change")
    area_percent: float = Field(..., description="Percentage of footprint affected within this region")
    description: str = Field(..., description="Human-readable description of the dynamic alteration")


class ChangeResult(BaseModel):
    """Structured output from ChangeEngine V3."""
    changed: bool = Field(..., description="True if surface alterations exceed threshold")
    summary: str = Field(..., description="Natural language summary of bi-temporal change analysis")
    change_area_percent: float = Field(..., description="Total footprint percentage exhibiting significant change")
    change_regions: List[ChangeRegion] = Field(default_factory=list, description="Localized bounding regions of identified changes")
    change_types: List[ChangeType] = Field(default_factory=list, description="List of change categories identified")
    change_map_url: Optional[str] = Field(None, description="Base64 encoded PNG of the change difference heatmap")
    t1_date: Optional[str] = Field(None, description="Acquisition date/timestamp of baseline image T1")
    t2_date: Optional[str] = Field(None, description="Acquisition date/timestamp of comparison image T2")
    method: str = Field(..., description="Change detection algorithm (e.g., 'spectral_differencing_v3', 'siam_change_net')")
    confidence: Optional[float] = Field(None, description="Confidence score for change detection")
    provenance: ModelProvenance = Field(ModelProvenance.REAL, description="Change engine provenance")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Image dimensions and temporal metadata")
