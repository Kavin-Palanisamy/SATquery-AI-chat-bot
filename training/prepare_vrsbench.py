"""
Prepares VRSBench (Visual Reasoning, Grounding & Captioning) dataset for fine-tuning.
"""
import argparse
import json
from pathlib import Path


def prepare_vrsbench(vrsbench_dir: str, output_manifest: str):
    src_path = Path(vrsbench_dir)
    out_path = Path(output_manifest)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Scanning '{vrsbench_dir}' for VRSBench grounding and VQA annotations...")
    samples = []
    
    ann_file = src_path / "vrsbench_annotations.json"
    if ann_file.exists():
        with open(ann_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            for item in data:
                samples.append({
                    "sample_id": item.get("id"),
                    "image_path": str(src_path / "images" / item.get("image_name", "")),
                    "caption": item.get("caption", ""),
                    "grounding": item.get("grounding", []),
                    "vqa_pairs": item.get("vqa", []),
                })

    with open(out_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    print(f"Generated VRSBench training manifest with {len(samples)} samples: {output_manifest}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="data/raw/vrsbench", help="VRSBench source directory")
    parser.add_argument("--output", type=str, default="data/processed/vrsbench_train.jsonl", help="Output JSONL")
    args = parser.parse_args()
    prepare_vrsbench(args.source, args.output)
