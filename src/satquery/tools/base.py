from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from satquery.schemas.models import ModelProvenance


class BaseTool(ABC):
    """Abstract base class for all specialist tools in SatQuery AI."""

    def __init__(
        self,
        name: str,
        version: str = "v3",
        supported_modalities: Optional[List[str]] = None,
        supported_tasks: Optional[List[str]] = None,
        provenance: ModelProvenance = ModelProvenance.REAL,
        description: Optional[str] = None,
    ):
        self.name = name
        self.version = version
        self.supported_modalities = supported_modalities or ["OPTICAL"]
        self.supported_tasks = supported_tasks or []
        self.provenance = provenance
        self.description = description

    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        """Executes the tool's specialist workflow."""
        pass

    def health(self) -> Dict[str, Any]:
        """Returns health telemetry for this tool."""
        return {
            "name": self.name,
            "version": self.version,
            "status": "HEALTHY",
            "provenance": self.provenance.value,
        }

    def metadata(self) -> Dict[str, Any]:
        """Returns tool capability metadata."""
        return {
            "name": self.name,
            "version": self.version,
            "supported_modalities": self.supported_modalities,
            "supported_tasks": self.supported_tasks,
            "provenance": self.provenance.value,
            "description": self.description,
        }
