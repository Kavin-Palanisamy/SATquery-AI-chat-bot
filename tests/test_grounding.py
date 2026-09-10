import numpy as np
import pytest
from PIL import Image

from satquery.agent.router import AgenticOrchestrator, analyze_query_intent
from satquery.analysis.grounding import (
    BaseGroundingModel,
    HeuristicGroundingModel,
    MockGroundingModel,
    RealGroundingModel,
    detect_grounding_regions,
    extract_target_entity,
    get_grounding_model,
    normalized_to_pixel_bbox,
    parse_vlm_bounding_boxes,
    pixel_to_normalized_bbox,
    pixel_to_percentage_bbox,
    validate_bounding_box,
    validate_grounding_boxes,
)
from satquery.schemas.vqa import AgentTaskType, BoundingBox


# ============================================================================
# TEST 1: Water Grounding
# ============================================================================
def test_1_water_grounding(sample_water_image):
    """TEST 1: Query 'Find and highlight the water body' -> GROUNDING, target='water body', valid bbox, no full image bbox."""
    query = "Find and highlight the water body in this image."
    intent = analyze_query_intent(query)
    assert intent.task == AgentTaskType.GROUNDING
    target = extract_target_entity(query)
    assert target == "water body"

    img = Image.open(sample_water_image)
    model = HeuristicGroundingModel()
    res = model.ground(img, query, target=target)

    assert res.success is True
    assert len(res.detections) > 0
    pbox = res.detections[0]
    # Verify not full image
    cov = (pbox.xmax - pbox.xmin) * (pbox.ymax - pbox.ymin)
    assert cov < 0.95
    assert cov > 0.01
    assert pbox.bbox_coverage_pct is not None
    assert pbox.mask_coverage_pct is not None
    assert res.mask_url is not None
    assert "data:image/png;base64," in res.mask_url


# ============================================================================
# TEST 2: Building Grounding
# ============================================================================
def test_2_building_grounding():
    """TEST 2: Query 'Find the buildings' -> target='buildings'."""
    query = "Find the buildings"
    intent = analyze_query_intent(query)
    assert intent.task == AgentTaskType.GROUNDING
    target = extract_target_entity(query)
    assert target in ("buildings", "built-up area")


# ============================================================================
# TEST 3: Road Grounding
# ============================================================================
def test_3_road_grounding():
    """TEST 3: Query 'Highlight the road' -> target='road'."""
    query = "Highlight the road"
    intent = analyze_query_intent(query)
    assert intent.task == AgentTaskType.GROUNDING
    target = extract_target_entity(query)
    assert target == "road"


# ============================================================================
# TEST 4: Vegetation Grounding
# ============================================================================
def test_4_vegetation_grounding():
    """TEST 4: Query 'Locate vegetation' -> target='vegetation'."""
    query = "Locate vegetation"
    intent = analyze_query_intent(query)
    assert intent.task == AgentTaskType.GROUNDING
    target = extract_target_entity(query)
    assert target in ("vegetation", "forest", "crops")


# ============================================================================
# TEST 5: No Target / Non-existent Target
# ============================================================================
def test_5_no_target_nonexistent(sample_water_image):
    """TEST 5: Query 'Find the moon' -> No fabricated detection, success=False."""
    query = "Find the moon"
    target = extract_target_entity(query)
    assert "moon" in target

    img = Image.open(sample_water_image)
    model = HeuristicGroundingModel()
    res = model.ground(img, query, target=target)
    assert res.success is False
    assert res.detections == []
    assert res.confidence is None
    assert "unable to reliably localize" in res.message.lower()


# ============================================================================
# TEST 6: Invalid Coordinates (Negative Coordinates)
# ============================================================================
def test_6_invalid_negative_coordinates():
    """TEST 6: Reject negative coordinates [-10, 20, 50, 80]."""
    is_valid, box, reason = validate_bounding_box(ymin=-0.10, xmin=0.20, ymax=0.50, xmax=0.80)
    assert is_valid is False
    assert box is None
    assert "negative" in reason.lower()


# ============================================================================
# TEST 7: Degenerate Bounding Box (x2 <= x1 or y2 <= y1)
# ============================================================================
def test_7_degenerate_bounding_box():
    """TEST 7: Reject degenerate coordinates where x2 <= x1 or y2 <= y1."""
    is_valid, box, reason = validate_bounding_box(ymin=0.50, xmin=0.50, ymax=0.50, xmax=0.80)
    assert is_valid is False
    assert box is None

    is_valid2, box2, reason2 = validate_bounding_box(ymin=0.20, xmin=0.80, ymax=0.60, xmax=0.30)
    assert is_valid2 is False
    assert box2 is None


# ============================================================================
# TEST 8: Full-Image Bounding Box Rejection
# ============================================================================
def test_8_full_image_bounding_box_rejected():
    """TEST 8: Reject full image bbox [0, 0, width, height] (>= 99.5% coverage)."""
    is_valid, box, reason = validate_bounding_box(ymin=0.0, xmin=0.0, ymax=1.0, xmax=1.0, max_coverage_threshold=0.995)
    assert is_valid is False
    assert box is None
    assert "full-image" in reason.lower()


# ============================================================================
# TEST 9: NaN / Infinite Coordinate Rejection
# ============================================================================
def test_9_nan_infinite_coordinates_rejected():
    """TEST 9: Reject NaN and infinite coordinate values."""
    is_valid_nan, box_nan, reason_nan = validate_bounding_box(ymin=float("nan"), xmin=0.1, ymax=0.8, xmax=0.9)
    assert is_valid_nan is False
    assert "nan" in reason_nan.lower()

    is_valid_inf, box_inf, reason_inf = validate_bounding_box(ymin=0.1, xmin=0.1, ymax=float("inf"), xmax=0.9)
    assert is_valid_inf is False
    assert "infinite" in reason_inf.lower() or "inf" in reason_inf.lower()


# ============================================================================
# TEST 10: Mask Validation & Separate Metrics
# ============================================================================
def test_10_mask_validation_and_coverage_separation(sample_water_image):
    """TEST 10: Segmentation mask generation, coverage calculation, and separation from bbox area."""
    img = Image.open(sample_water_image)
    model = HeuristicGroundingModel()
    res = model.ground(img, "Find water body", target="water body")

    assert res.success is True
    assert res.mask_url is not None
    assert res.mask_coverage_pct is not None
    assert res.bbox_coverage_pct is not None
    # Mask area must be <= Bounding box area
    assert res.mask_coverage_pct <= res.bbox_coverage_pct + 0.1
    assert "Bounding box covers" in res.message
    assert "Detected mask covers" in res.message


# ============================================================================
# TEST 11: Provenance Consistency
# ============================================================================
def test_11_provenance_consistency(sample_water_image):
    """TEST 11: RS-SpectralHeuristic-Grounder-v1 appears in trace, and mock-vlm-v1 is NOT present in heuristic trace."""
    img = Image.open(sample_water_image)
    orchestrator = AgenticOrchestrator()
    resp = orchestrator.execute("Find and highlight the water body in this image.", img)

    assert resp.task == "grounding"
    assert resp.tool_used == "RemoteSensingGroundingTool"
    assert resp.model == "RS-SpectralHeuristic-Grounder-v1"

    trace_str = str(resp.execution_trace)
    assert "RS-SpectralHeuristic-Grounder-v1" in trace_str
    assert "HEURISTIC" in trace_str
    # mock-vlm-v1 must NOT appear in heuristic grounding trace
    assert "mock-vlm-v1" not in trace_str

    # Check step 4 specifically
    step4 = [s for s in resp.execution_trace["steps"] if s["step"] == 4][0]
    assert "RS-SpectralHeuristic-Grounder-v1 [HEURISTIC]" in step4["details"]
    assert "mock-vlm-v1" not in step4["details"]


# ============================================================================
# TEST 12: VQA Task Regression Protection
# ============================================================================
def test_12_vqa_regression_preserved(sample_water_image):
    """TEST 12: Ensure VQA still works and routes correctly."""
    query = "Describe the land cover and major objects visible in this image."
    intent = analyze_query_intent(query)
    assert intent.task == AgentTaskType.VQA

    img = Image.open(sample_water_image)
    orchestrator = AgenticOrchestrator()
    resp = orchestrator.execute(query, img)
    assert resp.task == "vqa"
    assert resp.tool_used == "RemoteSensingVQATool"


# ============================================================================
# TEST 13: Bi-Temporal Change Regression Protection
# ============================================================================
def test_13_change_regression_preserved(sample_water_image, sample_urban_image):
    """TEST 13: Ensure change analysis still works."""
    query = "What changed between these two dates?"
    intent = analyze_query_intent(query)
    assert intent.task == AgentTaskType.BITEMPORAL_CHANGE

    img1 = Image.open(sample_water_image)
    img2 = Image.open(sample_urban_image)
    orchestrator = AgenticOrchestrator()
    resp = orchestrator.execute(query, img1, secondary_image=img2)
    assert resp.task == "bitemporal_change"
    assert resp.tool_used == "RemoteSensingChangeTool"


# ============================================================================
# TEST 14: Optical-SAR Fusion Regression Protection
# ============================================================================
def test_14_optical_sar_regression_preserved(sample_water_image, sample_sar_image):
    """TEST 14: Ensure Optical-SAR fusion still works."""
    query = "Use optical and SAR together to identify built-up areas."
    intent = analyze_query_intent(query)
    assert intent.task == AgentTaskType.OPTICAL_SAR_FUSION

    img_opt = Image.open(sample_water_image)
    img_sar = Image.open(sample_sar_image)
    orchestrator = AgenticOrchestrator()
    resp = orchestrator.execute(query, img_opt, secondary_image=img_sar, primary_filename="opt.png", secondary_filename="sar.png")
    assert resp.task == "optical_sar_fusion"
    assert resp.tool_used == "OpticalSARFusionTool"


# ============================================================================
# Additional Unit Tests for Coordinate Conversions & Hierarchy
# ============================================================================
def test_coordinate_conversion_utilities():
    """Verify pixel to normalized, normalized to pixel, and percentage conversions."""
    # 1. Pixel to normalized
    n_xmin, n_ymin, n_xmax, n_ymax = pixel_to_normalized_bbox(100, 50, 300, 200, 1000, 500)
    assert n_xmin == 0.1
    assert n_ymin == 0.1
    assert n_xmax == 0.3
    assert n_ymax == 0.4

    # 2. Normalized to pixel
    px1, py1, px2, py2 = normalized_to_pixel_bbox(0.1, 0.1, 0.3, 0.4, 1000, 500)
    assert px1 == 100
    assert py1 == 50
    assert px2 == 300
    assert py2 == 200

    # 3. Pixel to percentage
    pct = pixel_to_percentage_bbox(100, 50, 300, 200, 1000, 500)
    assert pct["x_range"] == "10%–30%"
    assert pct["y_range"] == "10%–40%"
    assert pct["xmin_pct"] == 10
    assert pct["xmax_pct"] == 30


def test_grounding_model_hierarchy():
    """Verify BaseGroundingModel, HeuristicGroundingModel, MockGroundingModel, RealGroundingModel."""
    mock_m = get_grounding_model(force_mock=True)
    assert isinstance(mock_m, MockGroundingModel)
    assert mock_m.model_type == "MOCK"

    heur_m = get_grounding_model(allow_heuristic=True)
    assert isinstance(heur_m, HeuristicGroundingModel)
    assert heur_m.model_type == "HEURISTIC"

    real_m = get_grounding_model(force_mock=False, allow_heuristic=False)
    assert isinstance(real_m, RealGroundingModel)
    assert real_m.model_type == "REAL"
