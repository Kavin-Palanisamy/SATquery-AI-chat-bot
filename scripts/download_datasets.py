"""
Remote Sensing Dataset Downloader (Phase 22).
Provides instructions and configurable download hooks for BigEarthNet, RSVQA, VRSBench, and CDVQA.
"""
import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Download Remote Sensing Datasets for SatQuery AI")
    parser.add_argument("--dataset", choices=["bigearthnet", "rsvqa", "vrsbench", "cdvqa", "all"], default="all", help="Target dataset to download")
    parser.add_argument("--output-dir", type=str, default="data/raw", help="Target directory for downloaded data")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== SatQuery AI Dataset Downloader ===")
    print(f"Selected Dataset: {args.dataset}")
    print(f"Output Directory: {out_dir.resolve()}")
    print("\nDataset Links & Instructions:")
    print("1. BigEarthNet-v1.0 / BigEarthNet.txt:")
    print("   Source: https://bigearth.net/ | https://zenodo.org/records/3894212")
    print("2. RSVQA (High Resolution & Low Resolution):")
    print("   Source: https://rsvqa.sylvainlobry.com/")
    print("3. VRSBench (Visual Reasoning, Grounding & Captioning):")
    print("   Source: https://github.com/NJU-GeomSense/VRSBench")
    print("4. CDVQA (Change Detection VQA):")
    print("   Source: https://github.com/Chen-Data-Lab/CDVQA")
    print("\n[NOTE] Large remote-sensing datasets are not downloaded automatically during server startup.")
    print("Use local cache directories or point environment variables to dataset roots.")


if __name__ == "__main__":
    main()
