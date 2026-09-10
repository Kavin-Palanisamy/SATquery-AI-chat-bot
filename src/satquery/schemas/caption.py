"""
Captioning and scene understanding result schemas.
"""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from satquery.schemas.evidence import EvidenceItem
from satquery.schemas.models import ModelProvenance


class CaptionResult(BaseModel):
    """Structured scene captioning output."""
    caption: str = Field(..., description="Qualitative scene narrative")
    land_cover_summary: List[str] = Field(default_factory=list, description="Major visible land-cover categories")
    visible_objects: List[str] = Field(default_factory=list, description="Identified prominent entities")
    limitations: List[str] = Field(default_factory=list, description="Analytical limitations of observation")
    confidence: Optional[float] = Field(None, description="Captioning confidence score")
    provenance: ModelProvenance = Field(ModelProvenance.MOCK, description="Model provenance")
    evidence: List[EvidenceItem] = Field(default_factory=list, description="Supporting visual evidence")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Inference telemetry")
