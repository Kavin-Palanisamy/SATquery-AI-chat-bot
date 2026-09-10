import numpy as np
import pytest
from PIL import Image

from satquery.analysis.crossmodal import analyze_optical_sar_pair


def test_analyze_optical_sar_pair():
    optical_img = Image.new("RGB", (100, 100), color=(100, 150, 80))
    sar_img = Image.new("L", (100, 100), color=190)

    answer, conf = analyze_optical_sar_pair(
        optical_img, sar_img, "Use optical and SAR to identify built-up areas"
    )
    assert "optical" in answer.lower() or "sar" in answer.lower() or "built-up" in answer.lower()
    assert 0.0 <= conf <= 1.0
