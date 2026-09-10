import numpy as np
import pytest
from PIL import Image

from satquery.analysis.change import analyze_bitemporal_change, generate_change_map


def test_generate_change_map():
    img1 = Image.new("RGB", (100, 100), color=(100, 100, 100))
    img2 = Image.new("RGB", (100, 100), color=(100, 100, 100))
    
    # Modify a patch in img2
    arr2 = np.array(img2)
    arr2[20:60, 20:60] = [220, 50, 50]
    img2 = Image.fromarray(arr2)

    diff, change_map_url = generate_change_map(img1, img2)
    assert diff.shape == (100, 100)
    assert change_map_url.startswith("data:image/png;base64,")


def test_analyze_bitemporal_change():
    img1 = Image.new("RGB", (128, 128), color=(50, 150, 50))
    img2 = Image.new("RGB", (128, 128), color=(50, 50, 200))

    answer, change_map_url, conf = analyze_bitemporal_change(
        img1, img2, "What changed between date T1 and T2 regarding water?"
    )
    assert "water" in answer.lower() or "change" in answer.lower()
    assert change_map_url.startswith("data:image/png;base64,")
    assert 0.0 <= conf <= 1.0
