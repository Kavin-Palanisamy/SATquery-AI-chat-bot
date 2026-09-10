from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from satquery.schemas.models import ModelProvenance


class FusedRegion(BaseModel):
    """Specific region analyzed through cross-sensor optical and SAR synergy."""
    region_id: int = Field(..., description="Region identifier")
    feature: str = Field(..., description="Target feature (e.g., 'cloud_penetrated_urban', 'specular_water_surface')")
    optical_status: str = Field(..., description="Observation status under optical spectrum (e.g., 'obscured_by_cirrus_clouds')")
    sar_status: str = Field(..., description="Observation status under microwave radar (e.g., 'high_double_bounce_backscatter')")
    ymin: float = Field(..., description="Top boundary (0.0 to 1.0)")
    xmin: float = Field(..., description="Left boundary (0.0 to 1.0)")
    ymax: float = Field(..., description="Bottom boundary (0.0 to 1.0)")
    xmax: float = Field(..., description="Right boundary (0.0 to 1.0)")
    confidence: Optional[float] = Field(None, description="Confidence score for this cross-modal identification")


class OpticalSARFusionResult(BaseModel):
    """Structured output from OpticalSARFusionEngine V3."""
    success: bool = Field(..., description="True if fusion analysis succeeded")
    summary: str = Field(..., description="Natural language cross-modal synthesis")
    optical_evidence: List[str] = Field(default_factory=list, description="Findings derived from optical bands (RGB/NIR)")
    sar_evidence: List[str] = Field(default_factory=list, description="Findings derived from SAR radar backscatter (VV/VH)")
    fused_evidence: List[str] = Field(default_factory=list, description="Cross-modal complementary insights (e.g. all-weather penetration)")
    regions: List[FusedRegion] = Field(default_factory=list, description="Spatial candidate regions with cross-sensor details")
    optical_cloud_cover_pct: Optional[float] = Field(None, description="Estimated cloud obscuration in optical scene")
    sar_penetration_verified: bool = Field(True, description="True if microwave backscatter revealed features beneath clouds")
    fusion_map_url: Optional[str] = Field(None, description="Base64 encoded PNG overlay of cross-modal false-color composite")
    method: str = Field(..., description="Fusion methodology (e.g., 'cross_modal_synergy_engine_v3')")
    confidence: Optional[float] = Field(None, description="Confidence score for fusion result")
    provenance: ModelProvenance = Field(ModelProvenance.REAL, description="Fusion engine provenance")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Optical and SAR metadata (polarizations, bands, CRS)")
