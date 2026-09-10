from typing import Any, Dict, Optional, Tuple, Union
from PIL import Image
from satquery.geo.validator import PairValidationResult, ValidationResult, validate_image_pair, validate_input_image
from satquery.schemas.models import ModelProvenance
from satquery.tools.base import BaseTool


class InputValidationTool(BaseTool):
    """Tool for validating format, CRS, dimensions, and sensor modality of input imagery."""

    def __init__(self):
        super().__init__(
            name="InputValidationTool",
            version="v3",
            supported_modalities=["OPTICAL", "SAR", "MULTISPECTRAL"],
            supported_tasks=["input_validation"],
            provenance=ModelProvenance.REAL,
            description="Agentic tool validating geospatial image integrity, dimensions, CRS, and pair compatibility",
        )

    def execute(
        self,
        primary_image: Union[str, bytes, Image.Image],
        secondary_image: Optional[Union[str, bytes, Image.Image]] = None,
        workflow: str = "vqa",
    ) -> Union[ValidationResult, PairValidationResult]:
        if secondary_image is not None or workflow in ("bitemporal_change", "optical_sar_fusion"):
            return validate_image_pair(primary_image, secondary_image, workflow=workflow)
        val_res, _ = validate_input_image(primary_image)
        return val_res
