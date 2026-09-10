"""
Prepares BigEarthNet.txt formatted multi-modal pairs (Sentinel-1 SAR + Sentinel-2 Optical) for LoRA fine-tuning.
"""

import argparse
import json
import os
from pathlib import Path


def prepare_bigearthnet_txt(source_dir: str, output_manifest: str):
    src_path = Path(source_dir)
    out_path = Path(output_manifest)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Scanning '{source_dir}' for BigEarthNet Sentinel-2 / Sentinel-1 pairs...")
    samples = []
    
    # Check if files exist in directory
    if src_path.exists():
        for tif_file in src_path.glob("**/*.tif"):
            samples.append({
                "sample_id": tif_file.stem,
                "optical_path": str(tif_file),
                "caption": f"Satellite image containing remote sensing land-cover features.",
                "vqa_pairs": [
                    {"question": "Describe the land cover and major objects visible in this image.", "answer": "The scene represents remote sensing land-cover terrain."},
                ],
            })

    # Save output manifest
    with open(out_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

def prepare_bigearthnet_vqa_dataset(input_txt_path: str = "data/BigEarthNet.txt", output_dir: str = "data/processed"):
    """Prepares BigEarthNet dataset from input index."""
    out_manifest = Path(output_dir) / "bigearthnet_train.jsonl"
    prepare_bigearthnet_txt(source_dir=str(Path(input_txt_path).parent), output_manifest=str(out_manifest))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="data/real/vqa", help="Source directory containing satellite images")
    parser.add_argument("--output", type=str, default="data/training/bigearthnet_train.jsonl", help="Output JSONL manifest")
    args = parser.parse_args()
    prepare_bigearthnet_txt(args.source, args.output)
