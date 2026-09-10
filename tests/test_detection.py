import numpy as np
import pytest
from PIL import Image
from satquery.analysis.detection import DetectionEngine


def test_building_detection_counting(sample_urban_image):
    img = Image.open(sample_urban_image)
    engine = DetectionEngine()
    res = engine.detect(img, target_class="building")

    assert res.success is True
    assert res.total_count >= 1
    assert len(res.objects) == res.total_count
    for obj in res.objects:
        assert 0.0 <= obj.bbox.xmin < obj.bbox.xmax <= 1.0
        assert 0.0 <= obj.bbox.ymin < obj.bbox.ymax <= 1.0


def test_no_objects_detection():
    # Uniform dark background
    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    engine = DetectionEngine()
    res = engine.detect(img, target_class="building")

    assert res.success is False
    assert res.total_count == 0
    assert len(res.objects) == 0
