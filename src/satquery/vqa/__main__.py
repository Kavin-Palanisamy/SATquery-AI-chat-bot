import argparse
import sys
from pathlib import Path

from satquery.geo.image_loader import ImageValidationError
from satquery.vqa.inference import answer_question


def main():
    parser = argparse.ArgumentParser(
        prog="python -m satquery.vqa",
        description="SatQuery AI — Remote Sensing Visual Question Answering CLI",
    )
    parser.add_argument(
        "--image",
        "-i",
        required=True,
        type=str,
        help="Path to input satellite image (GeoTIFF, TIFF, PNG, JPEG)",
    )
    parser.add_argument(
        "--question",
        "-q",
        required=True,
        type=str,
        help="Natural language question about the satellite image",
    )
    parser.add_argument(
        "--model",
        "-m",
        default=None,
        type=str,
        help="Optional override for model ID (e.g. Qwen/Qwen2-VL-2B-Instruct)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Force lightweight mock model mode for testing",
    )

    args = parser.parse_args()

    try:
        result = answer_question(
            image_path=args.image,
            question=args.question,
            model_id=args.model,
            force_mock=args.mock,
        )

        image_name = Path(args.image).name
        print("\n" + "=" * 50)
        print("SatQuery AI — Inference Result")
        print("=" * 50)
        print(f"Image:     {image_name}")
        print(f"Question:  {args.question}")
        print(f"Model:     {result['model']}")
        print(f"Answer:    {result['answer']}")
        if result.get("execution_time_sec"):
            print(f"Latency:   {result['execution_time_sec']}s")
        print("=" * 50 + "\n")

    except FileNotFoundError as e:
        print(f"\n[Error] {e}", file=sys.stderr)
        sys.exit(1)
    except ImageValidationError as e:
        print(f"\n[Image Validation Error] {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n[Inference Error] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
