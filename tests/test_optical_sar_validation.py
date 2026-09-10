import io
from pathlib import Path
import numpy as np
import pytest
from PIL import Image

from satquery.agent.executor import AgentExecutor
from satquery.geo.validator import detect_modality, validate_image_pair, validate_input_image
from satquery.schemas.vqa import AgentTaskType


@pytest.fixture
def optical_png(temp_dir):
    p = temp_dir / "opt.png"
    arr = np.random.randint(50, 200, (100, 100, 3), dtype=np.uint8)
    Image.fromarray(arr).save(p)
    return p


@pytest.fixture
def sar_png(temp_dir):
    # SAR exported as a 3-channel or 1-channel PNG
    p = temp_dir / "radar_sar.png"
    arr = np.random.randint(10, 240, (100, 100), dtype=np.uint8)
    Image.fromarray(arr).convert("RGB").save(p)
    return p


@pytest.fixture
def optical_jpeg(temp_dir):
    p = temp_dir / "optical_scene.jpg"
    arr = np.random.randint(50, 200, (100, 100, 3), dtype=np.uint8)
    Image.fromarray(arr).save(p, format="JPEG")
    return p


@pytest.fixture
def sar_jpeg(temp_dir):
    p = temp_dir / "sar_amplitude.jpg"
    arr = np.random.randint(10, 240, (100, 100), dtype=np.uint8)
    Image.fromarray(arr).convert("RGB").save(p, format="JPEG")
    return p


@pytest.fixture
def cloudy_optical_png(temp_dir):
    p = temp_dir / "cloudy_optical.png"
    arr = np.random.randint(50, 150, (100, 100, 3), dtype=np.uint8)
    # Add bright clouds (230-255 in all channels) covering 30% of image
    arr[0:40, :, :] = 240
    Image.fromarray(arr).save(p)
    return p


@pytest.fixture
def sar_geotiff(temp_dir):
    p = temp_dir / "sentinel1_sar.tif"
    arr = np.random.randint(20, 220, (100, 100), dtype=np.uint8)
    img = Image.fromarray(arr, mode="L")
    img.save(p)
    return p


# ============================================================================
# CASE 1: Optical PNG + SAR PNG
# ============================================================================
def test_case_1_optical_png_and_sar_png(optical_png, sar_png):
    executor = AgentExecutor()
    res = executor.execute(
        query="Use optical and SAR together to identify built-up and water regions.",
        primary_image=optical_png,
        secondary_image=sar_png,
        task_override="optical_sar_fusion",
        primary_metadata={"slot_modality": "OPTICAL"},
        secondary_metadata={"slot_modality": "SAR"},
    )
    assert res.task == "optical_sar_fusion"
    assert res.tool_used == "OpticalSARFusionTool"
    assert res.model == "RS-CrossModalSynergy-v1"
    assert "Optical and SAR" in res.headline or "Optical and SAR" in res.answer
    assert len(res.evidence_items) > 0
    # Trace check
    trace = res.execution_trace
    assert trace is not None
    assert trace["final_status"] == "COMPLETED"
    assert any(s["tool"] == "OpticalSARFusionTool" for s in trace["steps"])


# ============================================================================
# CASE 2: Optical JPEG + SAR JPEG
# ============================================================================
def test_case_2_optical_jpeg_and_sar_jpeg(optical_jpeg, sar_jpeg):
    executor = AgentExecutor()
    res = executor.execute(
        query="Combine optical photo and radar JPEG to detect infrastructure.",
        primary_image=optical_jpeg,
        secondary_image=sar_jpeg,
        task_override="optical_sar_fusion",
        primary_metadata={"slot_modality": "OPTICAL"},
        secondary_metadata={"slot_modality": "SAR"},
    )
    assert res.task == "optical_sar_fusion"
    assert res.tool_used == "OpticalSARFusionTool"
    assert res.model == "RS-CrossModalSynergy-v1"
    assert res.execution_trace["final_status"] == "COMPLETED"


# ============================================================================
# CASE 3: Optical GeoTIFF + SAR GeoTIFF
# ============================================================================
def test_case_3_optical_geotiff_and_sar_geotiff(sample_geotiff_image, sar_geotiff):
    executor = AgentExecutor()
    res = executor.execute(
        query="Fuse optical spectral reflectance with SAR microwave radar.",
        primary_image=sample_geotiff_image,
        secondary_image=sar_geotiff,
        task_override="optical_sar_fusion",
        primary_metadata={"modality": "OPTICAL"},
        secondary_metadata={"modality": "SAR"},
    )
    assert res.task == "optical_sar_fusion"
    assert res.tool_used == "OpticalSARFusionTool"
    assert res.execution_trace["final_status"] == "COMPLETED"


# ============================================================================
# CASE 4: Cloudy Optical PNG + SAR PNG
# ============================================================================
def test_case_4_cloudy_optical_png_and_sar_png(cloudy_optical_png, sar_png):
    executor = AgentExecutor()
    res = executor.execute(
        query="Use optical and SAR together to identify built-up and water regions through clouds.",
        primary_image=cloudy_optical_png,
        secondary_image=sar_png,
        task_override="optical_sar_fusion",
        primary_metadata={"slot_modality": "OPTICAL"},
        secondary_metadata={"slot_modality": "SAR"},
    )
    assert res.task == "optical_sar_fusion"
    assert res.tool_used == "OpticalSARFusionTool"
    assert res.execution_trace["final_status"] == "COMPLETED"
    assert "cloud" in res.answer.lower() or "radar" in res.answer.lower() or "optical" in res.answer.lower()


# ============================================================================
# CASE 5: Optical + Optical (Rejected with clean message)
# ============================================================================
def test_case_5_optical_plus_optical_rejection(optical_png, optical_jpeg):
    executor = AgentExecutor()
    res = executor.execute(
        query="Use optical and SAR together to identify built-up and water regions.",
        primary_image=optical_png,
        secondary_image=optical_jpeg,
        task_override="optical_sar_fusion",
        primary_metadata={"modality": "OPTICAL", "user_modality": "OPTICAL"},
        secondary_metadata={"modality": "OPTICAL", "user_modality": "OPTICAL"},
    )
    assert res.task == "validation_error"
    assert res.tool_used == "InputValidationTool"
    assert "Please provide one optical image and one SAR/radar image for this analysis." in res.answer
    assert res.execution_trace["final_status"] == "REJECTED"


# ============================================================================
# CASE 6: Only Optical (Rejected with clean message)
# ============================================================================
def test_case_6_only_optical_rejection(optical_png):
    executor = AgentExecutor()
    res = executor.execute(
        query="Use optical and SAR together to identify built-up and water regions.",
        primary_image=optical_png,
        secondary_image=None,
        task_override="optical_sar_fusion",
    )
    assert res.task == "validation_error"
    assert res.tool_used == "InputValidationTool"
    assert "Please provide one optical image and one SAR/radar image for this analysis." in res.answer
    assert res.execution_trace["final_status"] == "REJECTED"


# ============================================================================
# HTTP API Integration Test: /agent/analyze with explicit slot modalities
# ============================================================================
def test_api_optical_sar_multipart_upload(optical_png, sar_png):
    from fastapi.testclient import TestClient
    from satquery.api.main import app

    with TestClient(app) as client:
        with open(optical_png, "rb") as f1, open(sar_png, "rb") as f2:
            response = client.post(
                "/agent/analyze",
                data={
                    "question": "Use optical and SAR together to identify built-up and water regions.",
                    "task_mode": "optical_sar_fusion",
                    "primary_modality": "OPTICAL",
                    "secondary_modality": "SAR",
                },
                files={
                    "image": ("cloudy_opt.png", f1, "image/png"),
                    "secondary_image": ("radar_backscatter.png", f2, "image/png"),
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["task"] == "optical_sar_fusion"
        assert data["tool_used"] == "OpticalSARFusionTool"
        assert data["model"] == "RS-CrossModalSynergy-v1"
        assert data["execution_trace"]["final_status"] == "COMPLETED"
        assert any(
            s["tool"] == "OpticalSARFusionTool" for s in data["execution_trace"]["steps"]
        )
