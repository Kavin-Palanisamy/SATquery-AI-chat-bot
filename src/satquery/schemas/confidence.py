from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class CalibrationStatus(str, Enum):
    CALIBRATED = "CALIBRATED"
    UNCALIBRATED = "UNCALIBRATED"
    HEURISTIC = "HEURISTIC"
    UNAVAILABLE = "UNAVAILABLE"


class ConfidenceRecord(BaseModel):
    """Calibrated or heuristic confidence provenance record."""
    value: Optional[float] = Field(None, description="Confidence score [0.0 - 1.0] if available, null if uncalibrated")
    source: str = Field(..., description="Methodology or model reporting this score (e.g., 'spectral_heuristic', 'calibrated_softmax')")
    calibration_status: CalibrationStatus = Field(CalibrationStatus.UNCALIBRATED, description="Calibration state of the score")
    description: Optional[str] = Field(None, description="Human-readable explanation of score calibration")
    components: Dict[str, Any] = Field(default_factory=dict, description="Component-level confidence metrics (routing, model, evidence)")
