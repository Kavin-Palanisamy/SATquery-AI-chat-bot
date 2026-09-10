import pytest
from PIL import Image

from satquery.agent.router import (
    AgenticOrchestrator,
    analyze_query_intent,
    detect_image_modality,
    extract_target_entity,
)
from satquery.schemas.vqa import AgentTaskType
from satquery.vqa.model import MockVQAModel


@pytest.fixture
def orchestrator():
    model = MockVQAModel(model_id="mock-vlm-v1")
    return AgenticOrchestrator(vqa_model=model)


# ============================================================================
# 1. Linguistic & Semantic Intent Unit Tests
# ============================================================================

def test_intent_vqa_describe_landcover_objects():
    """CRITICAL TEST: 'Describe the land cover and major objects visible in this image' -> VQA (NOT Grounding)."""
    intent = analyze_query_intent("Describe the land cover and major objects visible in this image.")
    assert intent.task == AgentTaskType.VQA
    assert intent.requires_spatial_evidence is False
    assert intent.requires_multiple_images is False


def test_intent_vqa_landcover_types():
    intent = analyze_query_intent("What land cover types are visible?")
    assert intent.task == AgentTaskType.VQA


def test_intent_vqa_presence_verification():
    intent = analyze_query_intent("Is there a water body?")
    assert intent.task == AgentTaskType.VQA
    assert intent.intent in ("semantic_presence_question", "presence_verification")


def test_intent_vqa_terrain_type():
    intent = analyze_query_intent("What type of terrain is present?")
    assert intent.task == AgentTaskType.VQA


def test_intent_vqa_general_scene_understanding():
    intent = analyze_query_intent("What is visible in this satellite image?")
    assert intent.task == AgentTaskType.VQA


def test_intent_grounding_where_is_water():
    """Where is the water body -> Grounding."""
    intent = analyze_query_intent("Where is the water body?")
    assert intent.task == AgentTaskType.GROUNDING
    assert intent.requires_spatial_evidence is True


def test_intent_grounding_highlight_water():
    intent = analyze_query_intent("Highlight the water body.")
    assert intent.task == AgentTaskType.GROUNDING


def test_intent_grounding_find_buildings():
    intent = analyze_query_intent("Find the buildings.")
    assert intent.task == AgentTaskType.GROUNDING
    assert intent.target in ("buildings", "built-up area")


def test_intent_grounding_show_me_agricultural_area():
    intent = analyze_query_intent("Show me the agricultural area.")
    assert intent.task == AgentTaskType.GROUNDING


def test_intent_grounding_locate_runway():
    intent = analyze_query_intent("Locate the runway.")
    assert intent.task == AgentTaskType.GROUNDING


def test_intent_grounding_draw_box_around_lake():
    intent = analyze_query_intent("Draw a box around the lake.")
    assert intent.task == AgentTaskType.GROUNDING


def test_intent_grounding_where_are_urban_areas():
    intent = analyze_query_intent("Where are the urban areas?")
    assert intent.task == AgentTaskType.GROUNDING


def test_intent_bitemporal_change_between_dates():
    intent = analyze_query_intent("What changed between these two dates?")
    assert intent.task == AgentTaskType.BITEMPORAL_CHANGE
    assert intent.requires_multiple_images is True


def test_intent_bitemporal_has_built_up_increased():
    intent = analyze_query_intent("Has the built-up area increased?")
    assert intent.task == AgentTaskType.BITEMPORAL_CHANGE
    assert intent.target == "built-up area"


def test_intent_bitemporal_has_vegetation_decreased():
    intent = analyze_query_intent("Has vegetation decreased?")
    assert intent.task == AgentTaskType.BITEMPORAL_CHANGE


def test_intent_bitemporal_flood_location():
    intent = analyze_query_intent("Where did the flood occur?")
    assert intent.task == AgentTaskType.BITEMPORAL_CHANGE


def test_intent_crossmodal_optical_sar_together():
    intent = analyze_query_intent("Use the optical and SAR images together to identify built-up areas.")
    assert intent.task == AgentTaskType.OPTICAL_SAR_FUSION
    assert intent.requires_multiple_images is True


def test_intent_crossmodal_both_sensors():
    intent = analyze_query_intent("Use both sensors to identify water-covered regions.")
    assert intent.task == AgentTaskType.OPTICAL_SAR_FUSION


# ============================================================================
# 2. Modality & Sensor Detection Tests
# ============================================================================

def test_detect_image_modality_optical():
    mod = detect_image_modality(None, "sample_optical_s2.png", {"sensor": "Sentinel-2"})
    assert mod == "optical"


def test_detect_image_modality_sar():
    mod = detect_image_modality(None, "sample_sar_s1.png", {"sensor": "Sentinel-1"})
    assert mod == "sar"


# ============================================================================
# 3. Router Task Classification & Execution Integration Tests
# ============================================================================

def test_router_classify_vqa_describe_scene(orchestrator):
    task = orchestrator.classify_task("Describe the land cover and major objects visible in this image.")
    assert task == AgentTaskType.VQA


def test_router_classify_vqa_landcover(orchestrator):
    task = orchestrator.classify_task("What land cover types are visible?")
    assert task == AgentTaskType.VQA


def test_router_classify_vqa_is_there_water(orchestrator):
    task = orchestrator.classify_task("Is there a water body?")
    assert task == AgentTaskType.VQA


def test_router_classify_grounding_where_is_water(orchestrator):
    task = orchestrator.classify_task("Where is the water body?")
    assert task == AgentTaskType.GROUNDING


def test_router_classify_grounding_highlight(orchestrator):
    task = orchestrator.classify_task("Highlight the water body.")
    assert task == AgentTaskType.GROUNDING


def test_router_classify_grounding_find_buildings(orchestrator):
    task = orchestrator.classify_task("Find the buildings.")
    assert task == AgentTaskType.GROUNDING


def test_router_classify_bitemporal_two_images(orchestrator):
    task = orchestrator.classify_task(
        "What changed between these two dates?",
        has_secondary_image=True,
        primary_filename="t1.png",
        secondary_filename="t2.png",
    )
    assert task == AgentTaskType.BITEMPORAL_CHANGE


def test_router_classify_crossmodal_fusion(orchestrator):
    task = orchestrator.classify_task(
        "Use optical and SAR images together to identify built-up areas.",
        has_secondary_image=True,
        primary_filename="opt.png",
        secondary_filename="sar.png",
    )
    assert task == AgentTaskType.OPTICAL_SAR_FUSION


def test_router_execute_grounding_workflow(orchestrator, sample_water_image):
    img = Image.open(sample_water_image)
    res = orchestrator.execute("Locate the water body", primary_image=img, primary_filename="patch.png")
    assert res.task == "grounding"
    assert res.boxes is not None
    assert len(res.boxes) > 0
    assert res.tool_used == "RemoteSensingGroundingTool"
    assert "execution_trace" in res.model_dump()
    assert res.execution_trace["validation_status"] == "PASSED"
    assert "confidence_breakdown" in res.execution_trace


def test_router_execute_bitemporal_workflow(orchestrator, sample_png_image):
    img1 = Image.open(sample_png_image)
    img2 = Image.open(sample_png_image)
    res = orchestrator.execute(
        "What changed between these observation dates?",
        primary_image=img1,
        secondary_image=img2,
        primary_filename="sample_t1.png",
        secondary_filename="sample_t2.png",
    )
    assert res.task == "bitemporal_change"
    assert res.change_map_url is not None
    assert res.tool_used == "RemoteSensingChangeTool"
    assert res.execution_trace["validation_status"] == "PASSED"


def test_router_execute_single_image_change_validation_error(orchestrator, sample_png_image):
    """Query asks for change, but only 1 image provided -> input requirement validation notice."""
    img = Image.open(sample_png_image)
    res = orchestrator.execute("What changed between these two dates?", primary_image=img, primary_filename="sample_t1.png")
    assert res.execution_trace["validation_status"] == "FAILED"
    assert "two spatially corresponding" in res.answer or "Input Requirement Notice" in res.answer
    assert res.execution_trace["tool_used"] == "InputValidationEngine"


def test_router_execute_optical_sar_fusion_workflow(orchestrator, sample_png_image):
    opt = Image.open(sample_png_image)
    sar = Image.open(sample_png_image).convert("L")
    res = orchestrator.execute(
        "Use the optical and SAR images together to identify built-up areas.",
        primary_image=opt,
        secondary_image=sar,
        primary_filename="sample_optical_s2.png",
        secondary_filename="sample_sar_s1.png",
    )
    assert res.task == "optical_sar_fusion"
    assert res.tool_used == "OpticalSARFusionTool"
    assert res.execution_trace["validation_status"] == "PASSED"


def test_router_execute_optical_sar_mismatched_modality_validation_error(orchestrator, sample_png_image):
    """Query asks for Optical+SAR fusion, but two optical images provided -> compatibility notice."""
    img1 = Image.open(sample_png_image)
    img2 = Image.open(sample_png_image)
    res = orchestrator.execute(
        "Use the optical and SAR images together to identify built-up areas.",
        primary_image=img1,
        secondary_image=img2,
        primary_filename="sample_optical_1.png",
        secondary_filename="sample_optical_2.png",
    )
    assert res.execution_trace["validation_status"] == "FAILED"
    assert "complementary optical and radar/SAR" in res.answer or "Input Requirement Notice" in res.answer

