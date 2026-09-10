from typing import Any, Dict, Optional
from PIL import Image
from satquery.models.registry import get_model_registry
from satquery.schemas.fusion import OpticalSARFusionResult
from satquery.schemas.models import ModelProvenance
from satquery.tools.base import BaseTool


class OpticalSARFusionTool(BaseTool):
    """Tool for cross-modal fusion reasoning over Optical + SAR image pairs."""

    def __init__(self):
        super().__init__(
            name="OpticalSARFusionTool",
            version="v3",
            supported_modalities=["OPTICAL", "SAR"],
            supported_tasks=["optical_sar_fusion", "crossmodal_fusion"],
            provenance=ModelProvenance.REAL,
            description="Agentic tool conducting cross-modal fusion combining optical reflectance and SAR radar backscatter",
        )

    def execute(
        self,
        optical_img: Image.Image,
        sar_img: Image.Image,
        query: str,
        optical_meta: Optional[Dict] = None,
        sar_meta: Optional[Dict] = None,
    ) -> OpticalSARFusionResult:
        registry = get_model_registry()
        model = registry.get_model("OPTICAL_SAR_FUSION")
        return model.fuse(optical_img, sar_img, query, optical_meta=optical_meta, sar_meta=sar_meta)
