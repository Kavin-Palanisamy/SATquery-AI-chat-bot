from pathlib import Path
import pytest
from PIL import Image

from satquery.geo.image_loader import (
    CorruptedImageError,
    GeoImage,
    ImageValidationError,
    UnsupportedFormatError,
    load_image,
    validate_image_path,
)


def test_validate_image_path_valid(sample_png_image):
    validated = validate_image_path(sample_png_image)
    assert validated == sample_png_image


def test_validate_image_path_nonexistent():
    with pytest.raises(FileNotFoundError, match="Image file not found"):
        validate_image_path("non_existent_file_path_xyz.tif")


def test_validate_image_path_unsupported(unsupported_format_file):
    with pytest.raises(UnsupportedFormatError, match="Unsupported image format"):
        validate_image_path(unsupported_format_file)


def test_load_png_image(sample_png_image):
    geo_img = load_image(sample_png_image)
    assert isinstance(geo_img, GeoImage)
    assert geo_img.shape == (3, 100, 100)
    assert isinstance(geo_img.pil_image, Image.Image)
    assert geo_img.metadata["count"] == 3
    assert geo_img.metadata["filename"] == "sample.png"


def test_load_geotiff_image(sample_geotiff_image):
    geo_img = load_image(sample_geotiff_image)
    assert isinstance(geo_img, GeoImage)
    assert geo_img.shape == (3, 120, 120)
    assert isinstance(geo_img.pil_image, Image.Image)
    assert geo_img.metadata["filename"] == "sample_geo.tif"
    if geo_img.is_geospatial:
        assert geo_img.crs is not None
        assert geo_img.transform is not None
        assert len(geo_img.transform) == 6


def test_load_corrupted_image(corrupted_image_file):
    with pytest.raises(CorruptedImageError):
        load_image(corrupted_image_file)
