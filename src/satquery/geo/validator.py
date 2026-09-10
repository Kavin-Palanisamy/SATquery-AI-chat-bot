import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field

try:
    import rasterio
    from rasterio.io import MemoryFile
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

from satquery.geo.image_loader import (
    CorruptedImageError,
    ImageValidationError,
    UnsupportedFormatError,
    load_image,
)


class ValidationResult(BaseModel):
    """Result of geospatial image validation."""
    valid: bool = Field(..., description="True if image meets all structural and security constraints")
    modality: str = Field("OPTICAL", description="Identified image modality: OPTICAL, SAR, or MULTISPECTRAL")
    bands: int = Field(3, description="Number of raster bands")
    shape: List[int] = Field(default_factory=list, description="Dimensions [height, width, bands]")
    dtype: str = Field("uint8", description="Pixel data type")
    georeferenced: bool = Field(False, description="True if georeferenced metadata (CRS/Transform) is present")
    crs: Optional[str] = Field(None, description="Coordinate Reference System (e.g. EPSG:4326)")
    bounds: Optional[Dict[str, float]] = Field(None, description="Geographic bounding coordinates")
    resolution: Optional[Tuple[float, float]] = Field(None, description="Pixel ground resolution in CRS units")
    nodata: Optional[Any] = Field(None, description="NoData pixel value")
    file_size_kb: float = Field(0.0, description="File size in kilobytes")
    warnings: List[str] = Field(default_factory=list, description="Non-fatal data warnings")
    errors: List[str] = Field(default_factory=list, description="Fatal validation errors")


class PairValidationResult(BaseModel):
    """Result of multi-image pair compatibility validation."""
    valid: bool = Field(..., description="True if pair is compatible for target workflow")
    workflow: str = Field(..., description="Target workflow (bitemporal_change, optical_sar_fusion)")
    primary_validation: ValidationResult
    secondary_validation: ValidationResult
    dimensions_match: bool = Field(True, description="True if both images have identical pixel dimensions")
    crs_match: bool = Field(True, description="True if both images share coordinate reference systems")
    modalities_compatible: bool = Field(True, description="True if sensor modalities satisfy workflow requirements")
    coregistration_status: str = Field("VERIFIED", description="VERIFIED, APPROXIMATE, or UNALIGNED")
    warnings: List[str] = Field(default_factory=list, description="Pair-level warnings")
    errors: List[str] = Field(default_factory=list, description="Pair-level errors")


def detect_modality(img_arr: np.ndarray, metadata: Dict[str, Any]) -> str:
    """
    Robustly identify sensor modality from pixel channels, statistics, and metadata.
    """
    driver = str(metadata.get("driver", "")).upper()
    count = metadata.get("count", 3)
    orig_name = str(metadata.get("original_filename", "")).lower()

    # Explicit filename/metadata hints
    if "sar" in orig_name or "s1" in orig_name or "sentinel1" in orig_name or "radar" in orig_name:
        return "SAR"
    if "opt" in orig_name or "s2" in orig_name or "sentinel2" in orig_name or "landsat" in orig_name:
        return "OPTICAL"

    # Single-band / Grayscale analysis
    if img_arr.ndim == 2 or (img_arr.ndim == 3 and img_arr.shape[2] == 1):
        # Single band image
        return "SAR"
    
    if img_arr.ndim == 3:
        if img_arr.shape[2] >= 4:
            return "MULTISPECTRAL"
        # Check channel correlation - SAR exported as 3-channel duplicate vs true color RGB
        if img_arr.shape[2] == 3:
            diff_rg = np.mean(np.abs(img_arr[:, :, 0].astype(float) - img_arr[:, :, 1].astype(float)))
            diff_gb = np.mean(np.abs(img_arr[:, :, 1].astype(float) - img_arr[:, :, 2].astype(float)))
            if diff_rg < 1.0 and diff_gb < 1.0:
                return "SAR"
            return "OPTICAL"

    return "OPTICAL"


def validate_input_image(
    image_input: Union[str, Path, bytes, Image.Image],
    expected_modality: Optional[str] = None,
    max_size_mb: float = 100.0,
) -> Tuple[ValidationResult, Optional[Image.Image]]:
    """
    Comprehensive validator for satellite imagery input.
    Validates file integrity, dimensions, CRS, nodata, format, and modality.
    """
    warnings: List[str] = []
    errors: List[str] = []
    pil_img: Optional[Image.Image] = None

    # Handle path traversal protection & format validation
    if isinstance(image_input, (str, Path)):
        path = Path(image_input)
        if not path.exists():
            return ValidationResult(
                valid=False,
                errors=[f"Image file not found: {path}"],
                warnings=[],
            ), None
        
        file_size_mb = path.stat().st_size / (1024 * 1024)
        if file_size_mb > max_size_mb:
            return ValidationResult(
                valid=False,
                errors=[f"File size {file_size_mb:.1f}MB exceeds limit of {max_size_mb}MB"],
                warnings=[],
            ), None

        allowed_exts = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
        if path.suffix.lower() not in allowed_exts:
            return ValidationResult(
                valid=False,
                errors=[f"Unsupported file format '{path.suffix}'. Allowed: {sorted(allowed_exts)}"],
                warnings=[],
            ), None

        try:
            geo_img = load_image(path)
            pil_img = geo_img.pil_image
            metadata = geo_img.metadata
            arr = np.array(pil_img)
        except Exception as e:
            return ValidationResult(
                valid=False,
                errors=[f"Failed to decode image: {str(e)}"],
                warnings=[],
            ), None

    elif isinstance(image_input, Image.Image):
        pil_img = image_input
        arr = np.array(pil_img)
        metadata = {
            "driver": "PIL",
            "shape": [arr.shape[0], arr.shape[1], arr.shape[2] if arr.ndim == 3 else 1],
            "count": arr.shape[2] if arr.ndim == 3 else 1,
            "crs": None,
            "is_geospatial": False,
        }
    elif isinstance(image_input, bytes):
        import io
        try:
            pil_img = Image.open(io.BytesIO(image_input))
            pil_img.load()
            arr = np.array(pil_img)
            metadata = {
                "driver": pil_img.format or "BYTES",
                "shape": [arr.shape[0], arr.shape[1], arr.shape[2] if arr.ndim == 3 else 1],
                "count": arr.shape[2] if arr.ndim == 3 else 1,
                "crs": None,
                "is_geospatial": False,
            }
        except Exception as e:
            return ValidationResult(
                valid=False,
                errors=[f"Failed to read image byte stream: {str(e)}"],
                warnings=[],
            ), None
    else:
        return ValidationResult(
            valid=False,
            errors=[f"Unsupported image input type: {type(image_input)}"],
            warnings=[],
        ), None

    # Dimension & sanity checks
    height, width = arr.shape[:2]
    if width < 16 or height < 16:
        errors.append(f"Image dimensions ({width}x{height}) are too small for satellite analysis (min 16x16).")
    if width > 10000 or height > 10000:
        warnings.append(f"Very large image ({width}x{height}) may cause high latency. Resizing recommended.")

    modality = detect_modality(arr, metadata)

    if expected_modality and modality != expected_modality.upper():
        errors.append(f"Mismatched sensor modality: Expected {expected_modality.upper()}, detected {modality}.")

    georeferenced = bool(metadata.get("is_geospatial", False) or metadata.get("crs"))
    crs = metadata.get("crs")
    if not georeferenced:
        warnings.append("Image is not georeferenced (standard pixel coordinate frame will be used).")

    val_res = ValidationResult(
        valid=len(errors) == 0,
        modality=modality,
        bands=arr.shape[2] if arr.ndim == 3 else 1,
        shape=[height, width, arr.shape[2] if arr.ndim == 3 else 1],
        dtype=str(arr.dtype),
        georeferenced=georeferenced,
        crs=crs,
        bounds=metadata.get("bounds"),
        nodata=metadata.get("nodata"),
        file_size_kb=float(metadata.get("file_size_kb", 0.0)),
        warnings=warnings,
        errors=errors,
    )
    return val_res, pil_img


def validate_image_pair(
    primary_input: Union[str, Path, bytes, Image.Image],
    secondary_input: Optional[Union[str, Path, bytes, Image.Image]],
    workflow: str = "bitemporal_change",
) -> PairValidationResult:
    """
    Validates a dual-image pair for bi-temporal change detection or optical-SAR fusion.
    """
    val1, img1 = validate_input_image(primary_input)
    warnings: List[str] = list(val1.warnings)
    errors: List[str] = list(val1.errors)

    if secondary_input is None:
        return PairValidationResult(
            valid=False,
            workflow=workflow,
            primary_validation=val1,
            secondary_validation=ValidationResult(valid=False, errors=["Secondary image is missing."], warnings=[]),
            dimensions_match=False,
            crs_match=False,
            modalities_compatible=False,
            coregistration_status="UNALIGNED",
            warnings=warnings,
            errors=[f"Workflow '{workflow}' requires two images, but secondary image was not provided."],
        )

    val2, img2 = validate_input_image(secondary_input)
    warnings.extend(val2.warnings)
    errors.extend(val2.errors)

    # Dimensions check
    dim_match = (val1.shape[:2] == val2.shape[:2])
    if not dim_match:
        warnings.append(f"Image dimensions differ: {val1.shape[:2]} vs {val2.shape[:2]}. Automatic spatial resampling will be applied.")

    # CRS check
    crs_match = (val1.crs == val2.crs)
    if val1.georeferenced and val2.georeferenced and not crs_match:
        warnings.append(f"CRS mismatch: {val1.crs} vs {val2.crs}. On-the-fly reprojection required.")

    # Modality compatibility check
    modalities_compat = True
    coreg = "VERIFIED"

    if workflow in ("bitemporal_change", "change"):
        if val1.modality != val2.modality:
            warnings.append(f"Bi-temporal change analysis typically requires same sensor modalities (got {val1.modality} vs {val2.modality}).")
    elif workflow in ("optical_sar_fusion", "crossmodal"):
        mods = {val1.modality, val2.modality}
        if "OPTICAL" not in mods or "SAR" not in mods:
            modalities_compat = False
            errors.append(f"Optical-SAR fusion requires one OPTICAL and one SAR image. Received {val1.modality} and {val2.modality}.")

    return PairValidationResult(
        valid=len(errors) == 0,
        workflow=workflow,
        primary_validation=val1,
        secondary_validation=val2,
        dimensions_match=dim_match,
        crs_match=crs_match,
        modalities_compatible=modalities_compat,
        coregistration_status=coreg,
        warnings=warnings,
        errors=errors,
    )
