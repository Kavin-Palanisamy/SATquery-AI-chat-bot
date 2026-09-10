import pytest
from PIL import Image
import numpy as np

from satquery.agent.executor import AgentExecutor
from satquery.models.registry import get_model_registry
from satquery.tools.vqa import RemoteSensingVQATool


def test_vqa_scene_description_scenarios(sample_water_image, sample_urban_image):
    executor = AgentExecutor()

    # 1. Water image scene description
    res_water = executor.execute(
        query="Describe this image.",
        primary_image=sample_water_image,
        task_override="vqa",
    )
    assert res_water.task == "vqa"
    assert "water" in res_water.answer.lower()
    assert res_water.headline is not None
    assert "%" not in res_water.answer

    # 2. Urban image scene description
    res_urban = executor.execute(
        query="What can you see here?",
        primary_image=sample_urban_image,
        task_override="vqa",
    )
    assert res_urban.task == "vqa"
    assert "built-up" in res_urban.answer.lower() or "vegetation" in res_urban.answer.lower() or "roads" in res_urban.answer.lower()
    assert "%" not in res_urban.answer


def test_vqa_land_cover_types(sample_water_image):
    executor = AgentExecutor()
    res = executor.execute(
        query="What types of land cover are visible?",
        primary_image=sample_water_image,
        task_override="vqa",
    )
    assert res.task == "vqa"
    assert "land cover types" in res.answer.lower()
    assert "water" in res.answer.lower()
    assert "%" not in res.answer


def test_vqa_feature_presence_yes_no(sample_water_image, sample_urban_image):
    executor = AgentExecutor()

    # Presence in water image
    res_water_yes = executor.execute(
        query="Are there any water bodies in this image?",
        primary_image=sample_water_image,
        task_override="vqa",
    )
    assert "yes" in res_water_yes.answer.lower()
    assert "water" in res_water_yes.answer.lower()

    # Absence of water in urban image
    res_water_no = executor.execute(
        query="Are there any water bodies in this image?",
        primary_image=sample_urban_image,
        task_override="vqa",
    )
    assert "no" in res_water_no.answer.lower() or "not detected" in res_water_no.answer.lower()

    # Presence of buildings in urban image
    res_urban_yes = executor.execute(
        query="Are there buildings in this scene?",
        primary_image=sample_urban_image,
        task_override="vqa",
    )
    assert "yes" in res_urban_yes.answer.lower() or "built-up" in res_urban_yes.answer.lower()


def test_vqa_spatial_understanding(sample_water_image):
    executor = AgentExecutor()
    res = executor.execute(
        query="Where is the water body?",
        primary_image=sample_water_image,
        task_override="vqa",
    )
    assert res.task == "vqa"
    assert "water" in res.answer.lower()
    assert ("located" in res.answer.lower() or "section" in res.answer.lower() or "area" in res.answer.lower() or "portion" in res.answer.lower())


def test_vqa_comparative_urban_vs_rural(sample_urban_image, sample_water_image):
    executor = AgentExecutor()

    res_rural = executor.execute(
        query="Is this area mainly urban or rural?",
        primary_image=sample_water_image,
        task_override="vqa",
    )
    assert "rural" in res_rural.answer.lower() or "agricultural" in res_rural.answer.lower()

    res_urban = executor.execute(
        query="Is this area predominantly urban or rural?",
        primary_image=sample_urban_image,
        task_override="vqa",
    )
    assert "urban" in res_urban.answer.lower() or "rural" in res_urban.answer.lower()


def test_vqa_strict_hallucination_control(sample_water_image):
    executor = AgentExecutor()

    # Percentage query in VQA mode without segmentation evidence must refuse
    res_pct = executor.execute(
        query="What percentage of the image is water?",
        primary_image=sample_water_image,
        task_override="vqa",
    )
    assert "cannot reliably calculate" in res_pct.answer.lower()
    assert "%" not in res_pct.answer

    # Object count query in VQA mode without detection evidence must refuse
    res_cnt = executor.execute(
        query="How many buildings are in this scene?",
        primary_image=sample_water_image,
        task_override="vqa",
    )
    assert "cannot reliably determine" in res_cnt.answer.lower()

    # Exact area query in VQA mode must refuse
    res_area = executor.execute(
        query="What is the exact area of the water body in sq km?",
        primary_image=sample_water_image,
        task_override="vqa",
    )
    assert "cannot reliably calculate" in res_area.answer.lower()


def test_vqa_truthful_provenance(sample_water_image):
    registry = get_model_registry()
    vlm = registry.get_model("VQA")
    assert vlm is not None
    info = vlm.get_info()
    assert info.provenance.value == "MOCK" or info.provenance.value == "REAL"
    assert info.adapter is None or info.adapter == "LoRA"
