import pytest
from satquery.models.registry import get_model_registry
from satquery.schemas.models import ModelProvenance


def test_model_registry_catalog():
    reg = get_model_registry()
    models = reg.list_models()
    assert len(models) >= 5

    tasks = {m.task for m in models}
    assert "VQA" in tasks
    assert "GROUNDING" in tasks
    assert "SEGMENTATION" in tasks
    assert "DETECTION" in tasks
    assert "CHANGE" in tasks
    assert "OPTICAL_SAR_FUSION" in tasks


def test_model_registry_provenance():
    reg = get_model_registry()
    vqa_info = reg.get_model_info("Qwen2-VL-RS")
    assert vqa_info is not None
    assert vqa_info.provenance in (ModelProvenance.MOCK, ModelProvenance.REAL, ModelProvenance.ADAPTED)

    grounding_info = reg.get_model_info("RS-SpectralHeuristic-Grounder-v1")
    assert grounding_info is not None
    assert grounding_info.provenance == ModelProvenance.HEURISTIC


def test_system_capabilities():
    reg = get_model_registry()
    cap = reg.get_capabilities()
    assert cap["vqa"] is True
    assert cap["grounding"] is True
    assert cap["segmentation"] is True
    assert cap["detection"] is True
    assert cap["change"] is True
    assert cap["optical_sar"] is True
