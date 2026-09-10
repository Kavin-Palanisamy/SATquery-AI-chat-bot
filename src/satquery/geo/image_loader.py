from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image

try:
    import rasterio
    from rasterio.errors import RasterioIOError
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

from satquery.utils.logging import get_logger

logger = get_logger("satquery.geo.loader")

SUPPORTED_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".webp"}


class ImageValidationError(Exception):
    """Raised when an image fails validation checks."""
    pass


class UnsupportedFormatError(ImageValidationError):
    """Raised when the image format is not supported."""
    pass


class CorruptedImageError(ImageValidationError):
    """Raised when the image data is corrupted or cannot be decoded."""
    pass


@dataclass
class GeoImage:
    """Standardized image container for Remote Sensing VQA pipelines."""
    data: np.ndarray  # Raw array (C, H, W) or (H, W)
    pil_image: Image.Image  # Standard 8-bit RGB image for VLM consumption
    metadata: Dict[str, Any]  # Complete extracted metadata
    crs: Optional[str] = None  # CRS as WKT or EPSG
    transform: Optional[List[float]] = None  # 6-element affine transform
    shape: Optional[Tuple[int, ...]] = None  # (C, H, W)
    is_geospatial: bool = False  # Flag for georeferenced raster


def _normalize_to_rgb(arr: np.ndarray, nodata: Optional[float] = None) -> Image.Image:
    """
    Converts multi-band/16-bit raster arrays to an 8-bit RGB PIL Image using
    robust percentile contrast stretching (2% - 98%).
    """
    if arr.ndim == 2:
        # Grayscale single band (H, W) -> expand to (1, H, W)
        arr = arr[np.newaxis, ...]
    
    c, h, w = arr.shape
    if c == 0 or h == 0 or w == 0:
        raise ImageValidationError(f"Invalid image dimensions: shape={arr.shape}")

    # Select RGB bands
    if c == 1:
        # Replicate grayscale to 3-channel
        selected_bands = np.repeat(arr, 3, axis=0)
    elif c >= 3:
        # Take first 3 bands (e.g. Red, Green, Blue)
        selected_bands = arr[:3, :, :]
    else:
        # 2-band -> pad with average to make 3-channel
        avg_band = np.mean(arr, axis=0, keepdims=True)
        selected_bands = np.concatenate([arr, avg_band], axis=0)

    rgb_channels = []
    for band_idx in range(3):
        band = selected_bands[band_idx].astype(np.float32)

        # Mask nodata if specified
        if nodata is not None:
            valid_mask = band != nodata
        else:
            valid_mask = np.isfinite(band)

        if np.any(valid_mask):
            valid_pixels = band[valid_mask]
            p_min = np.percentile(valid_pixels, 2)
            p_max = np.percentile(valid_pixels, 98)

            if p_max > p_min:
                stretched = (band - p_min) / (p_max - p_min) * 255.0
            else:
                stretched = band - p_min
            
            clipped = np.clip(stretched, 0, 255).astype(np.uint8)
        else:
            clipped = np.zeros((h, w), dtype=np.uint8)

        rgb_channels.append(clipped)

    # Stack to (H, W, 3)
    rgb_arr = np.stack(rgb_channels, axis=-1)
    return Image.fromarray(rgb_arr, mode="RGB")


def validate_image_path(image_path: Union[str, Path]) -> Path:
    """Validates existence and format of an image path."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: '{image_path}'")
    
    if not path.is_file():
        raise ImageValidationError(f"Path is not a regular file: '{image_path}'")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFormatError(
            f"Unsupported image format: '{ext}'. Supported formats: {sorted(list(SUPPORTED_EXTENSIONS))}"
        )

    return path


def load_image(image_path: Union[str, Path]) -> GeoImage:
    """
    Loads and inspects a satellite image (GeoTIFF, TIFF, PNG, JPEG).
    Extracts geospatial metadata when available and converts it to a standard
    representation for Vision-Language Models.
    """
    path = validate_image_path(image_path)
    ext = path.suffix.lower()
    is_tiff = ext in {".tif", ".tiff"}

    if is_tiff and HAS_RASTERIO:
        try:
            with rasterio.open(path) as src:
                data = src.read()  # (bands, H, W)
                band_count = src.count
                if band_count == 0:
                    raise ImageValidationError("GeoTIFF contains no readable bands.")

                crs_str = str(src.crs) if src.crs else None
                transform_list = list(src.transform)[:6] if src.transform else None
                is_geo = bool(src.crs or src.transform)

                bounds_dict = {
                    "left": float(src.bounds.left),
                    "bottom": float(src.bounds.bottom),
                    "right": float(src.bounds.right),
                    "top": float(src.bounds.top),
                } if src.bounds else None

                metadata: Dict[str, Any] = {
                    "filename": path.name,
                    "driver": src.driver,
                    "count": band_count,
                    "shape": [int(band_count), int(src.height), int(src.width)],
                    "dtypes": [str(d) for d in src.dtypes],
                    "nodata": float(src.nodata) if src.nodata is not None else None,
                    "crs": crs_str,
                    "transform": transform_list,
                    "bounds": bounds_dict,
                    "is_geospatial": is_geo,
                }

                pil_img = _normalize_to_rgb(data, nodata=src.nodata)

                return GeoImage(
                    data=data,
                    pil_image=pil_img,
                    metadata=metadata,
                    crs=crs_str,
                    transform=transform_list,
                    shape=(band_count, src.height, src.width),
                    is_geospatial=is_geo,
                )
        except RasterioIOError as e:
            raise CorruptedImageError(f"Image could not be read or is corrupted: {e}")
        except Exception as e:
            if isinstance(e, ImageValidationError):
                raise
            raise CorruptedImageError(f"Failed reading GeoTIFF image: {e}")

    # Standard PIL fallback for PNG/JPEG or when rasterio is not available
    try:
        with Image.open(path) as img:
            img.verify()
        
        # Re-open after verify()
        with Image.open(path) as img:
            pil_rgb = img.convert("RGB")
            np_arr = np.array(pil_rgb)  # (H, W, 3)
            data = np.transpose(np_arr, (2, 0, 1))  # (3, H, W)
            h, w = img.size[1], img.size[0]

            metadata = {
                "filename": path.name,
                "driver": img.format or ext.replace(".", "").upper(),
                "count": 3,
                "shape": [3, h, w],
                "dtypes": [str(np_arr.dtype)],
                "nodata": None,
                "crs": None,
                "transform": None,
                "bounds": None,
                "is_geospatial": False,
            }

            return GeoImage(
                data=data,
                pil_image=pil_rgb,
                metadata=metadata,
                crs=None,
                transform=None,
                shape=(3, h, w),
                is_geospatial=False,
            )
    except Exception as e:
        raise CorruptedImageError(f"Image could not be read: {e}")
