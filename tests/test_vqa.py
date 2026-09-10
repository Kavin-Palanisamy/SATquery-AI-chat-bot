import pytest
from PIL import Image
from satquery.vqa.inference import answer_question
from satquery.vqa.model import (
    BaseVQAModel,
    MockVQAModel,
    RealVQAModel,
    get_vqa_model,
    load_model_config,
)
from satquery.vqa.adapters import RemoteSensingAdapter
from satquery.schemas.vqa import EvidenceItem, EvidenceType, AgentTaskType
from satquery.agent.router import AgenticOrchestrator


def test_model_config_loading():
    cfg = load_model_config()
    assert "selected_model" in cfg
    assert "candidates" in cfg
    assert len(cfg["candidates"]) >= 2


def test_mock_model_generation(sample_png_image):
    model = get_vqa_model(force_mock=True)
    assert isinstance(model, MockVQAModel)
    assert model.mode == "MOCK"
    assert model.is_remote_sensing_adapted is False
    assert model.rs_adaptation_status == "NOT LOADED"

    result = answer_question(
        image_path=sample_png_image,
        question="What is visible in this image?",
        model=model,
    )

    assert "answer" in result
    assert "model" in result
    assert result["model"] == "mock-vlm-v1"
    assert result["model_mode"] == "MOCK"
    assert result["rs_adaptation"] == "NOT LOADED"
    assert result["confidence"] is None  # Confidence remains null for uncalibrated prototype
    assert "metadata" in result
    assert result["metadata"]["filename"] == "sample.png"


# -----------------------------------------------------------------------------
# STEP 11: Dedicated Acceptance Tests
# -----------------------------------------------------------------------------

def test_1_vqa_presence_no_fabricated_percentage(sample_png_image):
    """TEST 1: Query 'Are there water bodies in this image?' -> task == VQA, mode == MOCK, no fabricated percentages."""
    img = Image.open(sample_png_image)
    orchestrator = AgenticOrchestrator()
    res = orchestrator.execute(
        query="Are there water bodies in this image?",
        primary_image=img,
    )
    assert res.task == AgentTaskType.VQA.value
    assert res.model_mode == "MOCK"
    assert res.confidence is None
    # No fabricated percentage claims
    assert "18.2%" not in res.answer
    assert "%" not in res.answer
    assert "water" in res.answer.lower()


def test_2_vqa_scene_description_no_fabricated_percentages(sample_png_image):
    """TEST 2: Query 'Describe the land cover and major objects visible in this image.' -> task == VQA, no fabricated percentages."""
    img = Image.open(sample_png_image)
    orchestrator = AgenticOrchestrator()
    res = orchestrator.execute(
        query="Describe the land cover and major objects visible in this image.",
        primary_image=img,
    )
    assert res.task == AgentTaskType.VQA.value
    assert res.model_mode == "MOCK"
    assert res.confidence is None
    # Verify no fabricated precise percentages
    assert "18.2%" not in res.answer
    assert "36.7%" not in res.answer
    assert "46.0%" not in res.answer
    assert "16.2%" not in res.answer
    assert "%" not in res.answer
    assert "Quantitative land-cover composition" not in res.answer


def test_3_vqa_quantitative_percentage_abstention(sample_png_image):
    """TEST 3: Query 'What percentage of the image is water?' -> strictly abstains without segmentation evidence."""
    img = Image.open(sample_png_image)
    orchestrator = AgenticOrchestrator()
    res = orchestrator.execute(
        query="What percentage of the image is water?",
        primary_image=img,
    )
    assert res.task == AgentTaskType.VQA.value
    assert "cannot reliably calculate" in res.answer.lower()
    assert "%" not in res.answer


def test_3b_vqa_quantitative_percentage_with_segmentation_evidence(sample_png_image):
    """TEST 3b: When explicit segmentation evidence with pixel counts is provided, report measured ratio."""
    seg_context = {
        "segmentation": {
            "target": "water",
            "pixel_count": 1520,
            "valid_pixel_count": 10000,
        }
    }
    result = answer_question(
        image_path=sample_png_image,
        question="What percentage of the image is water?",
        context=seg_context,
        force_mock=True,
    )
    assert "15.2%" in result["answer"]
    assert "segmentation" in result["answer"].lower()
    assert "water_pixel_ratio" in result["answer"].lower()


def test_4_grounding_routing_reason_consistency(sample_png_image):
    """TEST 4: Query 'Find and highlight the water body.' -> task == GROUNDING, reason contains spatial_localization, NOT scene_description."""
    img = Image.open(sample_png_image)
    orchestrator = AgenticOrchestrator()
    res = orchestrator.execute(
        query="Find and highlight the water body.",
        primary_image=img,
    )
    assert res.task == AgentTaskType.GROUNDING.value
    routing_reason = res.execution_trace.get("routing_reason", "")
    intent = res.execution_trace.get("intent", {}).get("intent", "")
    assert "spatial_localization" in intent or "spatial" in routing_reason.lower() or "highlight" in routing_reason.lower()
    assert "scene_description" not in intent
    assert "scene_description" not in routing_reason.lower()


def test_5_no_cross_query_contamination(sample_png_image):
    """TEST 5: Sequential queries do not leak text between responses."""
    img = Image.open(sample_png_image)
    orchestrator = AgenticOrchestrator()

    res1 = orchestrator.execute(
        query="Describe the land cover and major objects visible in this image.",
        primary_image=img,
    )
    res2 = orchestrator.execute(
        query="Are there water bodies in this image?",
        primary_image=img,
    )

    assert "The scene appears to contain" in res1.answer or "mixture of" in res1.answer
    assert "Yes. A water body appears to be visible in the scene." in res2.answer or "Water-like regions appear" in res2.answer
    # Ensure answer 2 does not contain answer 1
    assert res1.answer not in res2.answer


def test_6_mock_vqa_confidence_null(sample_png_image):
    """TEST 6: Mock VQA confidence is null (uncalibrated)."""
    img = Image.open(sample_png_image)
    orchestrator = AgenticOrchestrator()
    res = orchestrator.execute(
        query="Are there water bodies in this image?",
        primary_image=img,
    )
    assert res.confidence is None
    assert res.execution_trace["confidence_breakdown"]["model_confidence"] is None


def test_object_count_refusal(sample_png_image):
    result = answer_question(
        image_path=sample_png_image,
        question="How many buildings are in this scene?",
        force_mock=True,
    )
    answer = result["answer"]
    assert "cannot reliably determine" in answer.lower() or "count" in answer.lower()
    assert "37 buildings" not in answer


def test_exact_area_refusal(sample_png_image):
    result = answer_question(
        image_path=sample_png_image,
        question="What is the exact area of water?",
        force_mock=True,
    )
    answer = result["answer"]
    assert "cannot reliably calculate the exact area" in answer.lower()


def test_vqa_geotiff_inference(sample_geotiff_image):
    result = answer_question(
        image_path=sample_geotiff_image,
        question="Describe the land cover and water bodies.",
        force_mock=True,
    )

    assert result["answer"] is not None
    assert len(result["answer"]) > 0
    assert result["metadata"]["filename"] == "sample_geo.tif"
    assert result["model_mode"] == "MOCK"
    assert result["rs_adaptation"] == "NOT LOADED"


def test_vqa_empty_question_raises(sample_png_image):
    with pytest.raises(ValueError, match="Question cannot be empty"):
        answer_question(
            image_path=sample_png_image,
            question="   ",
            force_mock=True,
        )


def test_rs_adapter_not_loaded():
    adapter = RemoteSensingAdapter()
    assert adapter.is_adapted is False
    assert adapter.status_label == "NOT LOADED"
    assert "pipeline ready" in adapter.description.lower()


def test_real_vqa_model_graceful_fallback(sample_png_image):
    real_model = RealVQAModel(model_id="Qwen/Qwen2-VL-2B-Instruct", rs_adapter_path=None)
    assert real_model.mode == "MOCK"
    assert real_model.is_remote_sensing_adapted is False
    assert real_model.rs_adaptation_status == "NOT LOADED"

    result = real_model.answer(
        image=sample_png_image,
        question="Are there water bodies in this image?",
    )
    assert result.answer is not None
    assert result.model_mode == "MOCK"
    assert result.rs_adaptation == "NOT LOADED"
