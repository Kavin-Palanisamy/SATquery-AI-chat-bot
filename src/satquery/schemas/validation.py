"""
Input validation schemas for single and multi-modal remote sensing imagery.
"""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class InputMetadata(BaseModel):
    """Metadata describing a validated remote sensing input raster."""
    filename: Optional[str] = Field(None, description="Original filename")
    format: str = Field(..., description="Raster format (e.g. 'GEOTIFF', 'PNG', 'JPEG')")
    dimensions: List[int] = Field(..., description="[Height, Width] pixel dimensions")
    bands: int = Field(..., description="Number of spectral/raster channels")
    dtype: str = Field(..., description="Data type of raster pixels (e.g., 'uint8', 'uint16', 'float32')")
    modality: str = Field("OPTICAL", description="Derived sensor modality (OPTICAL, SAR, MULTISPECTRAL)")
    crs: Optional[str] = Field(None, description="Coordinate Reference System string (e.g., 'EPSG:4326')")
    georeferenced: bool = Field(False, description="True if raster contains valid geospatial affine transform")
    bounds: Optional[List[float]] = Field(None, description="Bounding coordinate box [minx, miny, maxx, maxy]")
    nodata: Optional[float] = Field(None, description="NoData pixel fill value")


class ValidationResult(BaseModel):
    """Structured result returned by InputValidationEngine."""
    valid: bool = Field(..., description="True if input passes all validation checks")
    modality: str = Field(..., description="Assessed image modality (OPTICAL, SAR, MULTISPECTRAL, UNKNOWN)")
    bands: int = Field(..., description="Number of channels")
    dimensions: Optional[List[int]] = Field(None, description="[Height, Width]")
    georeferenced: bool = Field(False, description="Whether input has georeferenced spatial tags")
    crs: Optional[str] = Field(None, description="Raster CRS if available")
    warnings: List[str] = Field(default_factory=list, description="Non-fatal validation warnings")
    errors: List[str] = Field(default_factory=list, description="Fatal input validation errors")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Detailed raster telemetry")
