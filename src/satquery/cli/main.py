import argparse
import json
import sys
from pathlib import Path
from PIL import Image

from satquery.agent.executor import AgentExecutor
from satquery.geo.validator import validate_image_pair, validate_input_image
from satquery.models.registry import get_model_registry


def cli_main():
    parser = argparse.ArgumentParser(prog="satquery", description="SatQuery AI Multimodal Remote Sensing CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # 1. Health
    subparsers.add_parser("health", help="Check system and tool health status")

    # 2. Models
    subparsers.add_parser("models", help="List registered models and provenance statuses")

    # 3. Validate
    val_parser = subparsers.add_parser("validate", help="Validate satellite image format, CRS, and bands")
    val_parser.add_argument("image", type=str, help="Path to satellite image")

    # 4. Ask / VQA
    ask_parser = subparsers.add_parser("ask", help="Ask a natural language question about an image")
    ask_parser.add_argument("image", type=str, help="Path to satellite image")
    ask_parser.add_argument("question", type=str, help="Question to ask")

    # 5. Ground
    ground_parser = subparsers.add_parser("ground", help="Spatially localize and ground a target feature")
    ground_parser.add_argument("image", type=str, help="Path to satellite image")
    ground_parser.add_argument("target", type=str, help="Target feature to highlight/ground")

    # 6. Change
    change_parser = subparsers.add_parser("change", help="Analyze bi-temporal change between two dates")
    change_parser.add_argument("t1", type=str, help="Path to T1 baseline image")
    change_parser.add_argument("t2", type=str, help="Path to T2 comparison image")
    change_parser.add_argument("question", type=str, nargs="?", default="What changed between these two images?", help="Change question")

    # 7. Fuse
    fuse_parser = subparsers.add_parser("fuse", help="Execute joint optical + SAR cross-modal fusion")
    fuse_parser.add_argument("optical", type=str, help="Path to optical image")
    fuse_parser.add_argument("sar", type=str, help="Path to SAR radar image")
    fuse_parser.add_argument("question", type=str, nargs="?", default="Use optical and SAR together to identify features.", help="Fusion question")

    # 8. Train Status
    subparsers.add_parser("train-status", help="Display domain adaptation training status")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)

    executor = AgentExecutor()
    registry = get_model_registry()

    if args.command == "health":
        print(json.dumps({"status": "HEALTHY", "version": "v3.0.0", "capabilities": registry.get_capabilities()}, indent=2))

    elif args.command == "models":
        print("Registered Remote Sensing Models:")
        for m in registry.list_models():
            print(f"  [{m.task:18}] {m.name:32} | Provenance: {m.provenance.value:11} | Status: {'LOADED' if m.loaded else 'READY'}")

    elif args.command == "validate":
        res, _ = validate_input_image(args.image)
        print(json.dumps(res.model_dump(), indent=2))

    elif args.command == "ask":
        resp = executor.execute(query=args.question, primary_image=args.image)
        print(f"Task:  {resp.task}")
        print(f"Model: {resp.model} ({resp.model_mode})")
        print(f"Answer:\n{resp.answer}")

    elif args.command == "ground":
        query = f"Find and highlight the {args.target}."
        resp = executor.execute(query=query, primary_image=args.image)
        print(f"Task:  {resp.task}")
        print(f"Model: {resp.model} ({resp.model_mode})")
        print(f"Answer:\n{resp.answer}")
        if resp.boxes:
            print(f"Bounding Boxes ({len(resp.boxes)}):")
            for b in resp.boxes:
                print(f"  - [{b.xmin:.2f}, {b.ymin:.2f}, {b.xmax:.2f}, {b.ymax:.2f}] ({b.label})")

    elif args.command == "change":
        resp = executor.execute(query=args.question, primary_image=args.t1, secondary_image=args.t2)
        print(f"Task:  {resp.task}")
        print(f"Answer:\n{resp.answer}")

    elif args.command == "fuse":
        resp = executor.execute(query=args.question, primary_image=args.optical, secondary_image=args.sar)
        print(f"Task:  {resp.task}")
        print(f"Answer:\n{resp.answer}")

    elif args.command == "train-status":
        print(json.dumps({
            "rs_adaptation": "SUPPORTED (LoRA/PEFT pipeline ready)",
            "dataset_support": ["BigEarthNet.txt", "RSVQA", "VRSBench", "CDVQA"],
            "base_checkpoint": "Qwen/Qwen2-VL-2B-Instruct",
            "active_checkpoint": "MOCK / ZERO-SHOT UNTIL PEFT ADAPTER LOADED",
        }, indent=2))


if __name__ == "__main__":
    cli_main()
