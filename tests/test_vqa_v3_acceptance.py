import pytest
from PIL import Image
from satquery.agent.executor import AgentExecutor
from satquery.schemas.intent import AgentTaskType


def test_acceptance_1_semantic_presence(sample_png_image):
    """TEST 1: Query 'Are there water bodies in this image?' -> Qualitative answer, no invented percentage."""
    executor = AgentExecutor()
    res = executor.execute(
        query="Are there water bodies in this image?",
        primary_image=sample_png_image,
    )
    assert res.task == "vqa"
    assert "18.2%" not in res.answer
    assert "%" not in res.answer
    assert res.confidence is None  # Uncalibrated


def test_acceptance_2_percentage_refusal_and_with_evidence(sample_png_image, sample_water_image):
    """TEST 2: Query 'What percentage of the image is water?' -> refusal on pure VQA without segmentation, percentage ONLY with segmentation evidence."""
    # 1. Direct VQA without segmentation evidence MUST refuse quantitative claims
    from satquery.tools.vqa import RemoteSensingVQATool
    vqa_tool = RemoteSensingVQATool()
    img = Image.open(sample_png_image)
    res_direct = vqa_tool.execute(img, "What percentage of the image is water?", context={})
    assert "cannot reliably calculate" in res_direct.answer.lower()
    assert "%" not in res_direct.answer

    # 2. Multi-step Agent with segmentation pipeline computes exact measured percentage
    executor = AgentExecutor()
    res_agent = executor.execute(
        query="What percentage of the image is water?",
        primary_image=sample_water_image,
    )
    assert "%" in res_agent.answer
    assert "based on verified segmentation evidence" in res_agent.answer.lower()
    assert res_agent.evidence_used == "SEGMENTATION"


def test_acceptance_3_grounding(sample_water_image):
    """TEST 3: Query 'Find and highlight the water body.' -> GROUNDING task, valid bbox, heuristic provenance."""
    executor = AgentExecutor()
    res = executor.execute(
        query="Find and highlight the water body.",
        primary_image=sample_water_image,
    )
    assert res.task == "grounding"
    assert res.model_mode == "HEURISTIC"
    assert res.boxes is not None and len(res.boxes) > 0
    box = res.boxes[0]
    assert 0.0 <= box.xmin < box.xmax <= 1.0
    assert 0.0 <= box.ymin < box.ymax <= 1.0


def test_acceptance_4_counting_refusal(sample_png_image):
    """TEST 4: Query 'How many buildings are visible?' -> No VQA hallucinated number like '37 buildings'."""
    executor = AgentExecutor()
    res = executor.execute(
        query="How many buildings are visible?",
        primary_image=sample_png_image,
    )
    assert "37 buildings" not in res.answer
    assert res.task == "vqa"


def test_acceptance_5_scene_description(sample_png_image):
    """TEST 5: Query 'Describe the land cover.' -> Qualitative scene description."""
    executor = AgentExecutor()
    res = executor.execute(
        query="Describe the land cover and major objects visible in this image.",
        primary_image=sample_png_image,
    )
    assert res.task == "vqa"
    assert "Quantitative land-cover composition" not in res.answer
    assert "%" not in res.answer


def test_acceptance_6_bitemporal_change(sample_png_image, sample_urban_image):
    """TEST 6: Query 'What changed between these images?' -> BITEMPORAL_CHANGE workflow."""
    executor = AgentExecutor()
    res = executor.execute(
        query="What changed between these two images?",
        primary_image=sample_png_image,
        secondary_image=sample_urban_image,
    )
    assert res.task == "bitemporal_change"
    assert res.change_map_url is not None


def test_acceptance_7_change_subtask_builtup(sample_png_image, sample_urban_image):
    """TEST 7: Query 'Has built-up area increased?' -> Change analysis + evidence."""
    executor = AgentExecutor()
    res = executor.execute(
        query="Has built-up area increased between T1 and T2?",
        primary_image=sample_png_image,
        secondary_image=sample_urban_image,
    )
    assert res.task == "bitemporal_change"
    assert "built-up" in res.answer.lower() or "surface" in res.answer.lower()


def test_acceptance_8_optical_sar_fusion(sample_png_image, sample_sar_image):
    """TEST 8: Query 'Use optical and SAR together' -> OPTICAL_SAR_FUSION."""
    executor = AgentExecutor()
    res = executor.execute(
        query="Use optical and SAR images together to identify features.",
        primary_image=sample_png_image,
        secondary_image=sample_sar_image,
    )
    assert res.task == "optical_sar_fusion"
    assert "radar" in res.answer.lower() or "backscatter" in res.answer.lower() or "optical" in res.answer.lower()


def test_acceptance_9_optical_sar_missing_sar_rejection(sample_png_image):
    """TEST 9: Single image with optical-SAR question -> VALIDATION_ERROR rejection."""
    executor = AgentExecutor()
    res = executor.execute(
        query="Use optical and SAR together to identify features.",
        primary_image=sample_png_image,
    )
    assert res.task == "validation_error"
    assert "requires two uploaded images" in res.answer.lower()


def test_acceptance_10_no_cross_query_contamination(sample_png_image):
    """TEST 10: Two sequential queries do not leak text between responses."""
    executor = AgentExecutor()
    res1 = executor.execute(
        query="Describe the land cover.",
        primary_image=sample_png_image,
    )
    res2 = executor.execute(
        query="Are there water bodies in this image?",
        primary_image=sample_png_image,
    )
    assert res1.answer != res2.answer
    assert res1.answer not in res2.answer


def test_acceptance_11_no_fabricated_percentage_without_segmentation(sample_png_image):
    """TEST 11: Run 'What percentage of the image is water?' with no segmentation evidence -> NO FABRICATED NUMBER."""
    from satquery.tools.vqa import RemoteSensingVQATool
    vqa_tool = RemoteSensingVQATool()
    img = Image.open(sample_png_image)
    res = vqa_tool.execute(img, "What percentage of the image is water?", context={})
    assert "cannot reliably calculate" in res.answer.lower()
    assert "%" not in res.answer


def test_acceptance_12_model_status_loaded_unloaded():
    """TEST 12: Check model status -> Expected actual loaded/unloaded status accurately reported."""
    from satquery.models.registry import get_model_registry
    registry = get_model_registry()
    vqa_model = registry.get_model("VQA")
    assert vqa_model is not None
    # RS adaptation is accurately reported as NOT LOADED until a real adapter checkpoint exists on disk
    assert vqa_model.rs_adaptation_status == "NOT LOADED"
    assert vqa_model.loaded is False  # Lazy loaded

