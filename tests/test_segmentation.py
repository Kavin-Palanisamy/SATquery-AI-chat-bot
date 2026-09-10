import numpy as np
import pytest
from PIL import Image
from satquery.analysis.segmentation import SegmentationEngine


def test_water_segmentation_area(sample_water_image):
    img = Image.open(sample_water_image)
    engine = SegmentationEngine()
    res = engine.segment(img, target_class="water")

    assert res.success is True
    assert res.target_class == "water"
    assert res.positive_pixels > 0
    assert res.valid_pixels == img.width * img.height
    assert 10.0 <= res.area_percent <= 30.0
    assert res.mask_url is not None
    assert "data:image/png;base64," in res.mask_url


def test_vegetation_segmentation(sample_png_image):
    # Create green image
    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    arr[:, :, 0] = 30
    arr[:, :, 1] = 180
    arr[:, :, 2] = 40
    img = Image.fromarray(arr)

    engine = SegmentationEngine()
    res = engine.segment(img, target_class="vegetation")
    assert res.success is True
    assert res.area_percent > 90.0


def test_no_target_segmentation(sample_png_image):
    # Blank black image
    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    img = Image.fromarray(arr)

    engine = SegmentationEngine()
    res = engine.segment(img, target_class="water")
    assert res.success is False
    assert res.positive_pixels == 0
    assert res.area_percent == 0.0
