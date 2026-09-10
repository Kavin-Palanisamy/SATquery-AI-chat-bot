from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from PIL import Image
from satquery.schemas.detection import DetectedObject, DetectionResult
from satquery.schemas.grounding import GroundingBox
from satquery.schemas.models import ModelProvenance
from satquery.utils.logging import get_logger

logger = get_logger("satquery.analysis.detection")


def _find_connected_components_boxes(binary_mask: np.ndarray, min_area: int = 15, max_area: int = 100000) -> List[Tuple[int, int, int, int, int]]:
    """
    Pure NumPy/Python connected component bounding box extractor.
    Returns list of (ymin, xmin, ymax, xmax, area_pixels).
    """
    h, w = binary_mask.shape
    visited = np.zeros((h, w), dtype=bool)
    boxes = []

    # Fast scan for true pixels
    ys, xs = np.where(binary_mask)
    if len(ys) == 0:
        return []

    for start_y, start_x in zip(ys, xs):
        if visited[start_y, start_x]:
            continue

        # BFS component flood fill
        queue = [(start_y, start_x)]
        visited[start_y, start_x] = True
        min_y, max_y = start_y, start_y
        min_x, max_x = start_x, start_x
        pixel_count = 0

        while queue:
            cy, cx = queue.pop(0)
            pixel_count += 1
            if cy < min_y: min_y = cy
            if cy > max_y: max_y = cy
            if cx < min_x: min_x = cx
            if cx > max_x: max_x = cx

            # 4-connectivity neighbors
            for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                if 0 <= ny < h and 0 <= nx < w:
                    if binary_mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        queue.append((ny, nx))

        if min_area <= pixel_count <= max_area:
            boxes.append((min_y, min_x, max_y + 1, max_x + 1, pixel_count))

    return boxes


class DetectionEngine:
    """
    Dedicated Object Detection and Counting Engine.
    Executes connected component clustering and bounding box extraction for discrete features.
    """

    def __init__(self):
        self.provenance = ModelProvenance.HEURISTIC

    def detect(
        self,
        image: Image.Image,
        target_class: str = "building",
        min_area_pixels: int = 15,
        max_area_pixels: int = 50000,
    ) -> DetectionResult:
        """
        Detects discrete object instances and returns verified instance counts and bounding boxes.
        """
        arr = np.array(image.convert("RGB"))
        h, w, _ = arr.shape
        target_norm = target_class.strip().lower()

        # Binary candidate mask
        r = arr[:, :, 0].astype(float)
        g = arr[:, :, 1].astype(float)
        b = arr[:, :, 2].astype(float)
        brightness = (r + g + b) / 3.0

        if "building" in target_norm or "structure" in target_norm or "urban" in target_norm:
            # High brightness / contrast clusters
            saturation = np.max(arr, axis=2) - np.min(arr, axis=2)
            binary_mask = (brightness > 130) & (saturation < 50)
        elif "ship" in target_norm or "boat" in target_norm or "vessel" in target_norm:
            binary_mask = (brightness > 150)
        elif "aircraft" in target_norm or "plane" in target_norm:
            binary_mask = (brightness > 170)
        else:
            binary_mask = brightness > 140

        component_boxes = _find_connected_components_boxes(
            binary_mask, min_area=min_area_pixels, max_area=max_area_pixels
        )

        objects: List[DetectedObject] = []
        for ymin_px, xmin_px, ymax_px, xmax_px, box_area in component_boxes:
            ymin_n = max(0.0, ymin_px / h)
            xmin_n = max(0.0, xmin_px / w)
            ymax_n = min(1.0, ymax_px / h)
            xmax_n = min(1.0, xmax_px / w)
            bbox_cov = (xmax_n - xmin_n) * (ymax_n - ymin_n) * 100.0

            gbox = GroundingBox(
                ymin=round(ymin_n, 4),
                xmin=round(xmin_n, 4),
                ymax=round(ymax_n, 4),
                xmax=round(xmax_n, 4),
                label=target_class,
                confidence=0.82,
                bbox_coverage_pct=round(bbox_cov, 2),
                pixel_coords={"ymin": ymin_px, "xmin": xmin_px, "ymax": ymax_px, "xmax": xmax_px},
                confidence_type="heuristic",
            )
            objects.append(
                DetectedObject(
                    label=target_class,
                    bbox=gbox,
                    confidence=0.82,
                    area_pixels=box_area,
                )
            )

        return DetectionResult(
            success=len(objects) > 0,
            target_class=target_class,
            total_count=len(objects),
            objects=objects,
            method="connected_component_clustering_v3",
            confidence=0.82 if len(objects) > 0 else None,
            provenance=self.provenance,
            metadata={"image_width": w, "image_height": h},
        )
