"""
Prepares RSVQA (Remote Sensing Visual Question Answering) dataset for instruction tuning.
"""
import argparse
import json
from pathlib import Path


def prepare_rsvqa(rsvqa_dir: str, output_manifest: str):
    src_path = Path(rsvqa_dir)
    out_path = Path(output_manifest)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Scanning '{rsvqa_dir}' for RSVQA questions and image pairs...")
    samples = []
    
    # Check for annotations file
    ann_file = src_path / "Questions.json"
    if ann_file.exists():
        with open(ann_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            for item in data.get("questions", []):
                samples.append({
                    "sample_id": item.get("id"),
                    "image_path": str(src_path / "Images" / item.get("image_name", "")),
                    "vqa_pairs": [
                        {"question": item.get("question"), "answer": item.get("answer", "")}
                    ]
                })

    with open(out_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    print(f"Generated RSVQA training manifest with {len(samples)} samples: {output_manifest}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="data/raw/rsvqa", help="RSVQA source directory")
    parser.add_argument("--output", type=str, default="data/processed/rsvqa_train.jsonl", help="Output JSONL")
    args = parser.parse_args()
    prepare_rsvqa(args.source, args.output)
