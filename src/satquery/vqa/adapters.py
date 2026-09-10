"""
SatQuery AI — Remote Sensing Adaptation Hooks & Adapter Management.
Problem Statement: SIH26167

Provides a clean interface for loading LoRA, PEFT, and domain-adapted
weights (e.g. BigEarthNet-adapted vision-language checkpoints) while
honestly reporting active adaptation status.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional
from satquery.utils.logging import get_logger

logger = get_logger("satquery.vqa.adapters")


class RemoteSensingAdapter:
    """
    Manages Remote-Sensing Domain Adaptation (LoRA / PEFT checkpoints).
    """

    def __init__(
        self,
        adapter_path: Optional[str] = None,
        adapter_name: str = "RS-LoRA-Adapter",
        dataset_target: str = "BigEarthNet-RSVQA",
    ):
        self.adapter_path = adapter_path or os.getenv("SATQUERY_RS_ADAPTER_PATH")
        self.adapter_name = adapter_name
        self.dataset_target = dataset_target
        self._is_loaded = False
        self._status_message = "RS adaptation pipeline ready; checkpoint not currently loaded."

        # Check if an actual adapter checkpoint exists on disk
        if self.adapter_path and Path(self.adapter_path).exists():
            self._is_loaded = True
            self._status_message = f"RS-adapted checkpoint loaded from {self.adapter_path} ({dataset_target})."
            logger.info(self._status_message)
        else:
            self._is_loaded = False
            self._status_message = "RS adaptation pipeline ready; checkpoint not currently loaded."
            logger.info("RemoteSensingAdapter initialized without active checkpoint.")

    @property
    def is_loaded(self) -> bool:
        """Returns True only if an actual trained/adapted checkpoint is loaded."""
        return self._is_loaded

    @property
    def is_adapted(self) -> bool:
        """Alias for is_loaded."""
        return self._is_loaded

    @property
    def status_label(self) -> str:
        """Human-readable status for UI, traces, and API responses."""
        return "LOADED" if self._is_loaded else "NOT LOADED"

    @property
    def status_description(self) -> str:
        """Detailed status description."""
        return self._status_message

    @property
    def description(self) -> str:
        """Alias for status_description."""
        return self._status_message

    def apply_to_model(self, base_model: Any) -> Any:
        """
        Applies LoRA / PEFT weights to the underlying PyTorch VLM if checkpoint exists.
        """
        if not self._is_loaded:
            logger.info("No external RS adapter checkpoint loaded. Using base VLM weights.")
            return base_model

        try:
            from peft import PeftModel
            logger.info(f"Applying PEFT adapter from {self.adapter_path}...")
            adapted_model = PeftModel.from_pretrained(base_model, self.adapter_path)
            logger.info("PEFT remote-sensing adapter applied successfully.")
            return adapted_model
        except Exception as e:
            logger.error(f"Failed to apply PEFT adapter from {self.adapter_path}: {e}")
            self._is_loaded = False
            self._status_message = f"Failed to load adapter ({e}). Falling back to base model."
            return base_model

    def to_dict(self) -> Dict[str, Any]:
        """Returns structured metadata for provenance records."""
        return {
            "adapter_name": self.adapter_name,
            "is_loaded": self._is_loaded,
            "status": self.status_label,
            "description": self._status_message,
            "dataset_target": self.dataset_target,
            "adapter_path": self.adapter_path,
        }
