from typing import Any, Dict, List, Optional
from satquery.tools.base import BaseTool
from satquery.tools.captioning import CaptioningTool
from satquery.tools.change import ChangeDetectionTool
from satquery.tools.detection import DetectionTool
from satquery.tools.grounding import GroundingTool
from satquery.tools.optical_sar import OpticalSARFusionTool
from satquery.tools.segmentation import SegmentationTool
from satquery.tools.validation import InputValidationTool
from satquery.tools.vqa import RemoteSensingVQATool
from satquery.utils.logging import get_logger

logger = get_logger("satquery.tools.registry")


class ToolRegistry:
    """
    Centralized Tool Registry providing tool discovery, metadata inspection, and dispatch.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ToolRegistry, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._tools: Dict[str, BaseTool] = {}
        self._init_default_tools()
        self._initialized = True

    def _init_default_tools(self):
        self.register_tool(InputValidationTool())
        self.register_tool(RemoteSensingVQATool())
        self.register_tool(CaptioningTool())
        self.register_tool(GroundingTool())
        self.register_tool(SegmentationTool())
        self.register_tool(DetectionTool())
        self.register_tool(ChangeDetectionTool())
        self.register_tool(OpticalSARFusionTool())

    def register_tool(self, tool: BaseTool):
        self._tools[tool.name] = tool
        logger.info(f"Registered Tool [{tool.name}] (v{tool.version}) for tasks: {tool.supported_tasks}")

    def get_tool(self, name: str) -> Optional[BaseTool]:
        return self._tools.get(name)

    def list_tools(self) -> List[Dict[str, Any]]:
        return [t.metadata() for t in self._tools.values()]


# Singleton tool registry instance
tool_registry = ToolRegistry()


def get_tool_registry() -> ToolRegistry:
    return tool_registry
