from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class EvidenceType(str, Enum):
    # V3 Standard Evidence Types
    TEXT = "text"
    BBOX = "bbox"
    MASK = "mask"
    DETECTION = "detection"
    CHANGE_MASK = "change_mask"
    STATISTIC = "statistic"
    METADATA = "metadata"
    MODEL_OUTPUT = "model_output"
    
    # Backward compatibility aliases
    SEMANTIC = "semantic"
    SPATIAL = "spatial"
    TEMPORAL = "temporal"
    CROSS_MODAL = "cross_modal"


class EvidenceItem(BaseModel):
    """Unified structured evidence item supporting evidence-aware VQA and auditing."""
    type: EvidenceType = Field(..., description="Evidence category (text, bbox, mask, detection, change_mask, statistic)")
    source: str = Field(..., description="Tool or model that generated this evidence (e.g., 'RS-SpectralHeuristic-Grounder-v1')")
    target: Optional[str] = Field(None, description="Identified entity, class, or feature of interest")
    bbox: Optional[Dict[str, Any]] = Field(None, description="Spatial coordinates if applicable [xmin, ymin, xmax, ymax]")
    mask_coverage_pct: Optional[float] = Field(None, description="Measured pixel mask coverage percentage")
    bbox_coverage_pct: Optional[float] = Field(None, description="Bounding box coverage percentage")
    area_pixels: Optional[int] = Field(None, description="Number of positive pixels in segmentation/detection mask")
    valid_pixels: Optional[int] = Field(None, description="Total number of valid (non-nodata) pixels in image footprint")
    count: Optional[int] = Field(None, description="Identified discrete object count")
    confidence: Optional[float] = Field(None, description="Confidence associated with this specific evidence item")
    description: str = Field(..., description="Human-auditable explanation of the verified evidence")
    provenance: Optional[str] = Field("HEURISTIC", description="Evidence provenance: REAL, ADAPTED, HEURISTIC, MOCK, UNAVAILABLE")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional technical metadata")
