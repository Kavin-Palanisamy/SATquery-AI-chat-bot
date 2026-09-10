"""
VQA Benchmark Evaluator (Phase 24).
Evaluates accuracy / exact match on RSVQA / BigEarthNet QA pairs.
"""
from typing import Any, Dict, List
from PIL import Image
from satquery.models.registry import get_model_registry


def evaluate_vqa(samples: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    registry = get_model_registry()
    model = registry.get_model("VQA")
    
    if not samples:
        return {
            "task": "VQA",
            "model": model.name if model else "None",
            "provenance": model.provenance.value if model else "UNAVAILABLE",
            "total_evaluated": 0,
            "accuracy": None,
            "status": "NO_EVALUATION_DATASET_LOADED",
            "note": "Real benchmark metrics require downloaded evaluation dataset."
        }
    
    correct = 0
    for s in samples:
        ans = model.answer_question(s["image"], s["question"]).answer.strip().lower()
        if ans == s["ground_truth"].strip().lower():
            correct += 1
            
    return {
        "task": "VQA",
        "model": model.name,
        "provenance": model.provenance.value,
        "total_evaluated": len(samples),
        "accuracy": round(correct / len(samples), 4),
        "status": "COMPLETED"
    }


if __name__ == "__main__":
    import json
    res = evaluate_vqa()
    print(json.dumps(res, indent=2))
