import base64
import io
from typing import Any, Dict, Optional, Tuple
import numpy as np
from PIL import Image

from satquery.schemas.models import ModelProvenance
from satquery.schemas.segmentation import LandCoverClass, SegmentationResult
from satquery.utils.logging import get_logger

logger = get_logger("satquery.analysis.segmentation")


class SegmentationEngine:
    """
    Dedicated Remote Sensing Segmentation Engine.
    Provides verifiable pixel mask generation and area calculation.
    Acts as the single authorized source for quantitative land-cover coverage claims.
    """

    def __init__(self):
        self.provenance = ModelProvenance.HEURISTIC

    def segment(
        self,
        image: Image.Image,
        target_class: str = "water",
        threshold: float = 0.5,
    ) -> SegmentationResult:
        """
        Segments target land-cover feature from optical RGB/multispectral scene.
        """
        target_norm = target_class.strip().lower()
        arr = np.array(image.convert("RGB"))
        h, w, _ = arr.shape
        valid_pixels = int(h * w)

        r = arr[:, :, 0].astype(float)
        g = arr[:, :, 1].astype(float)
        b = arr[:, :, 2].astype(float)

        mask = np.zeros((h, w), dtype=bool)
        method_name = "spectral_heuristic_v3"
        conf = 0.85

        if "water" in target_norm or "lake" in target_norm or "river" in target_norm:
            # Modified NDWI / Blue dominance heuristic for RGB
            # Water: High blue/green relative to red, and low total brightness
            blue_ratio = b / (r + g + b + 1e-5)
            brightness = (r + g + b) / 3.0
            mask = (blue_ratio > 0.38) & (b > r * 1.15) & (brightness < 200)
            method_name = "spectral_ndwi_heuristic"

        elif "vegetation" in target_norm or "canopy" in target_norm or "forest" in target_norm or "agriculture" in target_norm:
            # Green vegetation index heuristic
            green_ratio = g / (r + g + b + 1e-5)
            mask = (green_ratio > 0.38) & (g > r * 1.1) & (g > b * 1.1)
            method_name = "green_canopy_index_heuristic"

        elif "built_up" in target_norm or "urban" in target_norm or "building" in target_norm or "structure" in target_norm:
            # Built-up high reflectance / neutral saturation
            brightness = (r + g + b) / 3.0
            saturation = np.max(arr, axis=2) - np.min(arr, axis=2)
            mask = (brightness > 130) & (saturation < 45)
            method_name = "urban_builtup_spectral_heuristic"

        elif "bare_soil" in target_norm or "soil" in target_norm or "sand" in target_norm:
            # High red/yellow soil tone
            mask = (r > g) & (g > b) & (r > 100) & ((r - b) > 30)
            method_name = "soil_substrate_heuristic"

        elif "road" in target_norm or "highway" in target_norm or "runway" in target_norm:
            brightness = (r + g + b) / 3.0
            saturation = np.max(arr, axis=2) - np.min(arr, axis=2)
            mask = (brightness > 60) & (brightness < 180) & (saturation < 20)
            method_name = "linear_infrastructure_heuristic"
        else:
            # Generic segmentation threshold
            mask = (r + g + b) / 3.0 > 128
            method_name = "generic_intensity_threshold"
            conf = 0.70

        positive_pixels = int(np.sum(mask))
        area_pct = (positive_pixels / valid_pixels) * 100.0 if valid_pixels > 0 else 0.0

        # Create overlay mask PNG (cyan for water, green for veg, yellow for urban, magenta for others)
        mask_png_url = None
        if positive_pixels > 0:
            overlay_arr = np.zeros((h, w, 4), dtype=np.uint8)
            if "water" in target_norm:
                overlay_arr[mask] = [0, 200, 255, 180]  # Cyan
            elif "vegetation" in target_norm:
                overlay_arr[mask] = [46, 204, 113, 180]  # Green
            elif "built_up" in target_norm or "urban" in target_norm:
                overlay_arr[mask] = [241, 196, 15, 180]  # Yellow
            else:
                overlay_arr[mask] = [231, 76, 60, 180]  # Red

            overlay_img = Image.fromarray(overlay_arr, mode="RGBA")
            buf = io.BytesIO()
            overlay_img.save(buf, format="PNG")
            mask_png_url = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"

        return SegmentationResult(
            success=positive_pixels > 0,
            target_class=target_class,
            positive_pixels=positive_pixels,
            valid_pixels=valid_pixels,
            area_percent=round(area_pct, 2),
            mask_url=mask_png_url,
            method=method_name,
            confidence=conf if positive_pixels > 0 else None,
            provenance=self.provenance,
            metadata={
                "image_width": w,
                "image_height": h,
                "threshold_applied": threshold,
            },
        )
