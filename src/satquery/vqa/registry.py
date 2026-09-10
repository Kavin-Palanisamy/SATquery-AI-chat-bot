"""
SatQuery AI — VQA Model Registry & Factory.
Problem Statement: SIH26167
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml

from satquery.utils.logging import get_logger

logger = get_logger("satquery.vqa.registry")


def load_model_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Loads model configuration from YAML."""
    if not config_path:
        config_path = os.getenv("SATQUERY_MODEL_CONFIG", "configs/model.yaml")
    path = Path(config_path)
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.warning(f"Could not parse config at {config_path}: {e}")
            return {}
    return {}


class VQAModelRegistry:
    """Registry for available VQA model architectures and backends."""

    REGISTERED_MODELS = {
        "mock-vlm-v1": {
            "name": "Mock VLM Engine (VQA V2)",
            "backend": "mock",
            "default_device": "cpu",
            "is_mock": True,
        },
        "Qwen/Qwen2-VL-2B-Instruct": {
            "name": "Qwen2-VL 2B Instruct",
            "backend": "transformers",
            "architecture": "Qwen2VLForConditionalGeneration",
            "default_device": "auto",
            "is_mock": False,
        },
        "vikhyatk/moondream2": {
            "name": "Moondream 2",
            "backend": "transformers",
            "architecture": "MoondreamForConditionalGeneration",
            "default_device": "cpu",
            "is_mock": False,
        },
        "MBZUAI/geochat-7B": {
            "name": "GeoChat 7B Remote Sensing VLM",
            "backend": "remote_sensing",
            "architecture": "LLaVA-1.5-RS",
            "default_device": "cuda",
            "is_mock": False,
        },
    }

    @classmethod
    def get_info(cls, model_id: str) -> Dict[str, Any]:
        return cls.REGISTERED_MODELS.get(
            model_id,
            {"name": model_id, "backend": "transformers", "is_mock": False},
        )
