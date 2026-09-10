"""
Comprehensive verification script testing the 7 required queries in SatQuery AI (VQA V2).
"""

import os
from pathlib import Path
from PIL import Image
import numpy as np

from satquery.agent.router import AgenticOrchestrator
from satquery.vqa.model import get_vqa_model


def create_synthetic_scene(name="optical_sample.png", width=300, height=300):
    os.makedirs("data/samples", exist_ok=True)
    img_path = Path("data/samples") / name
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Background: green vegetation
    arr[:, :] = [45, 130, 45]
    
    # Western sector: water body (low red, high blue)
    arr[:, :120] = [20, 50, 160]
    
    # Eastern sector: urban built-up (grayish)
    arr[60:240, 180:280] = [140, 140, 145]
    
    img = Image.fromarray(arr)
    img.save(img_path)
    return img_path


def create_sar_scene(name="sar_sample.png", width=300, height=300):
    os.makedirs("data/samples", exist_ok=True)
    img_path = Path("data/samples") / name
    arr = np.zeros((height, width), dtype=np.uint8)
    # Low backscatter for water (dark)
    arr[:, :120] = 15
    # High backscatter for urban (bright double-bounce)
    arr[60:240, 180:280] = 230
    # Moderate backscatter for vegetation volume scattering
    arr[:, 120:180] = 85
    arr[:60, 180:] = 80
    arr[240:, 180:] = 80
    
    img = Image.fromarray(arr, mode="L")
    img.save(img_path)
    return img_path


def run_verification():
    opt_path = create_synthetic_scene("primary_scene.png")
    sar_path = create_sar_scene("secondary_sar_scene.png")
    t2_path = create_synthetic_scene("secondary_t2_scene.png")
    
    opt_img = Image.open(opt_path)
    sar_img = Image.open(sar_path)
    t2_img = Image.open(t2_path)
    
    vqa_model = get_vqa_model(force_mock=True)
    orchestrator = AgenticOrchestrator(vqa_model=vqa_model)
    
    test_queries = [
        {
            "id": 1,
            "query": "Are there water bodies in this image?",
            "img1": opt_img,
            "img2": None,
            "meta1": {"filename": "primary_scene.png", "modality": "optical"},
            "meta2": None,
        },
        {
            "id": 2,
            "query": "Describe the land cover and major objects visible in this image.",
            "img1": opt_img,
            "img2": None,
            "meta1": {"filename": "primary_scene.png", "modality": "optical"},
            "meta2": None,
        },
        {
            "id": 3,
            "query": "Where is the water body?",
            "img1": opt_img,
            "img2": None,
            "meta1": {"filename": "primary_scene.png", "modality": "optical"},
            "meta2": None,
        },
        {
            "id": 4,
            "query": "Find and highlight the water body.",
            "img1": opt_img,
            "img2": None,
            "meta1": {"filename": "primary_scene.png", "modality": "optical"},
            "meta2": None,
        },
        {
            "id": 5,
            "query": "What percentage of the image is water?",
            "img1": opt_img,
            "img2": None,
            "meta1": {"filename": "primary_scene.png", "modality": "optical"},
            "meta2": None,
        },
        {
            "id": 6,
            "query": "What changed between these two images?",
            "img1": opt_img,
            "img2": t2_img,
            "meta1": {"filename": "primary_scene_2022.png", "modality": "optical"},
            "meta2": {"filename": "secondary_scene_2024.png", "modality": "optical"},
        },
        {
            "id": 7,
            "query": "Use the optical and SAR images together to identify built-up and water-covered regions.",
            "img1": opt_img,
            "img2": sar_img,
            "meta1": {"filename": "sentinel2_optical.png", "modality": "optical"},
            "meta2": {"filename": "sentinel1_sar.png", "modality": "sar"},
        },
    ]

    print("=" * 80)
    print("SATQUERY AI — VQA V2 VERIFICATION (7 CORE QUERIES)")
    print("=" * 80)

    for item in test_queries:
        qid = item["id"]
        q = item["query"]
        res = orchestrator.execute(
            query=q,
            primary_image=item["img1"],
            secondary_image=item["img2"],
            primary_meta=item["meta1"],
            secondary_meta=item["meta2"],
            primary_filename=item["meta1"]["filename"] if item["meta1"] else "primary.png",
            secondary_filename=item["meta2"]["filename"] if item["meta2"] else None,
        )

        print(f"\n[{qid}] Query: \"{q}\"")
        print(f"  Selected Task: {res.task.upper()}")
        print(f"  Tool:          {res.tool_used}")
        print(f"  Model:         {res.model}")
        print(f"  Mode:          {res.model_mode}")
        print(f"  RS Adaptation: {res.rs_adaptation}")
        print(f"  Evidence:      {res.evidence_used}")
        print(f"  Confidence:    {'Not calibrated' if res.confidence is None else f'{res.confidence * 100:.1f}%'}")
        print(f"  Answer:        {res.answer}")
        if res.boxes:
            b = res.boxes[0]
            print(f"  Bounding Box:  [{b.xmin * 100:.1f}%-{b.xmax * 100:.1f}% X, {b.ymin * 100:.1f}%-{b.ymax * 100:.1f}% Y] (Coverage: {b.bbox_coverage_pct or 0}%)")
        print(f"  Execution Steps: {len(res.execution_trace.get('steps', []))} steps logged in trace.")


if __name__ == "__main__":
    run_verification()
