from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ModelProvenance(str, Enum):
    REAL = "REAL"
    ADAPTED = "ADAPTED"
    HEURISTIC = "HEURISTIC"
    MOCK = "MOCK"
    UNAVAILABLE = "UNAVAILABLE"


class ModelStatus(str, Enum):
    LOADED = "LOADED"
    NOT_LOADED = "NOT_LOADED"
    READY = "READY"
    UNAVAILABLE = "UNAVAILABLE"
    FALLBACK = "FALLBACK"


class ModelInfo(BaseModel):
    """Centralized Model Registry schema defining model capability and provenance."""
    name: str = Field(..., description="Unique model identifier (e.g., 'Qwen2-VL-RS', 'RS-SpectralHeuristic-Grounder-v1')")
    version: str = Field("v1", description="Model architecture or checkpoint release version")
    task: str = Field(..., description="Primary specialist task (VQA, GROUNDING, SEGMENTATION, DETECTION, CHANGE, OPTICAL_SAR)")
    modality: List[str] = Field(default_factory=lambda: ["OPTICAL"], description="Supported sensor modalities (OPTICAL, MULTISPECTRAL, SAR)")
    provenance: ModelProvenance = Field(..., description="Provenance level: REAL, ADAPTED, HEURISTIC, MOCK, UNAVAILABLE")
    checkpoint: Optional[str] = Field(None, description="Local or HuggingFace checkpoint path / identifier")
    adapter: Optional[str] = Field(None, description="Adapter type if active (e.g., 'LoRA', 'PEFT', 'None')")
    dataset: Optional[str] = Field(None, description="Dataset used for remote-sensing domain adaptation (e.g., 'BigEarthNet.txt', 'RSVQA')")
    loaded: bool = Field(False, description="True if active weights are in memory / GPU")
    device: str = Field("cpu", description="Execution target device (cpu, cuda, mps)")
    dtype: str = Field("float32", description="Inference tensor datatype (bfloat16, float16, float32)")
    description: Optional[str] = Field(None, description="Human-readable description of model specialization")
