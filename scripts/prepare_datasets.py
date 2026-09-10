"""
Dataset preparation driver (Phase 22).
Prepares raw datasets into instruction-tuning conversation format for RS-VLM fine-tuning.
"""
import argparse
import sys
from pathlib import Path
from training.prepare_bigearthnet_txt import prepare_bigearthnet_vqa_dataset


def main():
    parser = argparse.ArgumentParser(description="Prepare Remote Sensing Datasets for SatQuery AI")
    parser.add_argument("--dataset", choices=["bigearthnet", "rsvqa", "vrsbench", "cdvqa", "all"], default="bigearthnet", help="Dataset to prepare")
    parser.add_argument("--input-file", type=str, default="data/BigEarthNet.txt", help="Input dataset index or annotation file")
    parser.add_argument("--output-dir", type=str, default="data/processed", help="Output directory for processed JSONL pairs")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset in ("bigearthnet", "all"):
        print("Preparing BigEarthNet instruction dataset...")
        prepare_bigearthnet_vqa_dataset(input_txt_path=args.input_file, output_dir=str(out_dir))
    
    print(f"Preparation complete. Processed files saved in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
