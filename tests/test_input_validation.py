import pytest
from PIL import Image
from satquery.geo.validator import (
    detect_modality,
    validate_image_pair,
    validate_input_image,
)


def test_valid_png_validation(sample_png_image):
    val_res, img = validate_input_image(sample_png_image)
    assert val_res.valid is True
    assert val_res.modality in ("OPTICAL", "SAR")
    assert val_res.bands == 3
    assert len(val_res.errors) == 0


def test_valid_geotiff_validation(sample_geotiff_image):
    val_res, img = validate_input_image(sample_geotiff_image)
    assert val_res.valid is True
    assert len(val_res.errors) == 0


def test_corrupted_file_validation(corrupted_image_file):
    val_res, img = validate_input_image(corrupted_image_file)
    assert val_res.valid is False
    assert len(val_res.errors) > 0


def test_unsupported_format_validation(unsupported_format_file):
    val_res, img = validate_input_image(unsupported_format_file)
    assert val_res.valid is False
    assert "Unsupported file format" in val_res.errors[0]


def test_optical_sar_pair_validation_success(sample_png_image, sample_sar_image):
    pair_res = validate_image_pair(sample_png_image, sample_sar_image, workflow="optical_sar_fusion")
    assert pair_res.valid is True
    assert pair_res.modalities_compatible is True


def test_optical_sar_pair_validation_missing_sar(sample_png_image, sample_urban_image):
    pair_res = validate_image_pair(sample_png_image, sample_urban_image, workflow="optical_sar_fusion")
    assert pair_res.valid is False
    assert "requires one OPTICAL and one SAR image" in pair_res.errors[0]


def test_bitemporal_pair_validation_missing_secondary(sample_png_image):
    pair_res = validate_image_pair(sample_png_image, None, workflow="bitemporal_change")
    assert pair_res.valid is False
    assert "requires two images" in pair_res.errors[0]
