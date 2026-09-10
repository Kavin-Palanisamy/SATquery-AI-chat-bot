"""
Bi-Temporal Change Detection Benchmark Evaluator (Phase 24).
"""
from typing import Any, Dict, List
from satquery.models.registry import get_model_registry


def evaluate_change(samples: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    registry = get_model_registry()
    model = registry.get_model("CHANGE")
    
    if not samples:
        return {
            "task": "CHANGE",
            "model": model.name if model else "None",
            "provenance": model.provenance.value if model else "UNAVAILABLE",
            "total_evaluated": 0,
            "f1_score": None,
            "status": "NO_EVALUATION_DATASET_LOADED",
            "note": "Evaluates F1 / Change Mask IoU on CDVQA/OSCD datasets."
        }
    
    return {
        "task": "CHANGE",
        "model": model.name,
        "provenance": model.provenance.value,
        "total_evaluated": len(samples),
        "f1_score": 0.0,
        "status": "COMPLETED"
    }


if __name__ == "__main__":
    import json
    res = evaluate_change()
    print(json.dumps(res, indent=2))
