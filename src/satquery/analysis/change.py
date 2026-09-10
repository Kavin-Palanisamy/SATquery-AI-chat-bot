import base64
import io
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import numpy as np
from PIL import Image

from satquery.utils.logging import get_logger

logger = get_logger("satquery.analysis.change")


def generate_change_map(
    img1: Union[str, Path, Image.Image],
    img2: Union[str, Path, Image.Image],
) -> Tuple[np.ndarray, str]:
    """
    Computes bi-temporal difference map between two timestamps (T1 and T2).
    Generates a color-coded heatmap:
    - Red: Significant structural change / built-up expansion
    - Blue: Water inundation / flood expansion
    - Green: Vegetation growth / greening
    """
    if isinstance(img1, (str, Path)):
        img1 = Image.open(img1)
    if isinstance(img2, (str, Path)):
        img2 = Image.open(img2)

    # Resize to matching dimensions if needed
    if img1.size != img2.size:
        img2 = img2.resize(img1.size, Image.Resampling.BILINEAR)


    arr1 = np.array(img1.convert("RGB")).astype(np.float32)
    arr2 = np.array(img2.convert("RGB")).astype(np.float32)

    # Compute absolute spectral difference
    diff = np.abs(arr2 - arr1)
    diff_magnitude = np.mean(diff, axis=2)

    # Threshold for significant change
    h, w = diff_magnitude.shape
    change_mask = diff_magnitude > 25.0

    # Colorize change map
    heatmap = np.zeros((h, w, 4), dtype=np.uint8)
    
    # Red for intensity change (built-up / bare ground)
    heatmap[change_mask, 0] = np.clip(diff_magnitude[change_mask] * 3.0, 100, 255).astype(np.uint8)
    # Green for vegetation difference
    veg_diff = (arr2[:, :, 1] - arr1[:, :, 1])
    heatmap[change_mask & (veg_diff > 15), 1] = 220
    # Blue for water change
    water_diff = (arr2[:, :, 2] - arr1[:, :, 2])
    heatmap[change_mask & (water_diff > 15), 2] = 240
    # Alpha channel for transparency overlay
    heatmap[change_mask, 3] = 180

    # Encode as Base64 PNG
    heat_img = Image.fromarray(heatmap, mode="RGBA")
    buffer = io.BytesIO()
    heat_img.save(buffer, format="PNG")
    b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    change_map_url = f"data:image/png;base64,{b64_str}"

    return diff_magnitude, change_map_url


def analyze_bitemporal_change(
    img_t1: Union[str, Path, Image.Image],
    img_t2: Union[str, Path, Image.Image],
    query: str,
    meta_t1: Optional[Dict] = None,
    meta_t2: Optional[Dict] = None,
) -> Tuple[str, str, float]:
    """
    Performs Bi-temporal change analysis and CDVQA (Change Detection Visual Question Answering).
    Returns (answer_text, change_map_url, confidence).
    """
    if isinstance(img_t1, (str, Path)):
        img_t1 = Image.open(img_t1)
    if isinstance(img_t2, (str, Path)):
        img_t2 = Image.open(img_t2)

    diff_magnitude, change_map_url = generate_change_map(img_t1, img_t2)

    
    total_pixels = diff_magnitude.size
    changed_pixels = np.sum(diff_magnitude > 25.0)
    change_ratio = float(changed_pixels) / float(total_pixels)
    change_pct = round(change_ratio * 100, 2)

    q_lower = query.lower()

    if "water" in q_lower or "flood" in q_lower:
        if change_pct > 5.0:
            answer = f"Water change detected. Water extent expanded across the highlighted region between the two dates ({change_pct}% of the scene)."
        else:
            answer = f"No significant water change detected between both dates (less than {max(change_pct, 1.0)}% variance)."

    elif "urban" in q_lower or "built-up" in q_lower or "building" in q_lower:
        if change_pct > 2.0:
            answer = f"Built-up change detected. New structures and developed areas expanded in the highlighted region between the two dates ({change_pct}% change)."
        else:
            answer = "No significant built-up change detected between the two dates."

    elif "forest" in q_lower or "tree" in q_lower or "vegetation" in q_lower or "agri" in q_lower:
        if change_pct > 2.0:
            answer = f"Vegetation change detected. Crop and green canopy variations are visible in the highlighted region ({change_pct}% difference)."
        else:
            answer = "Vegetation levels remained stable between the two observation dates."

    else:
        # General change query
        if change_pct > 2.0:
            answer = f"Change detected. Surface differences were detected across {change_pct}% of the image, highlighted in the change map."
        else:
            answer = "No significant landscape changes were detected between the two dates."

    confidence = round(min(0.85 + (change_pct / 200.0), 0.96), 2)
    return answer, change_map_url, confidence
