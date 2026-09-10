from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class VRSBenchSample(BaseModel):
    sample_id: str
    image_path: str
    grounding_queries: List[Dict[str, Any]] = Field(default_factory=list)
    vqa_pairs: List[Dict[str, str]] = Field(default_factory=list)
    caption: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class VRSBenchAdapter:
    """Adapter for VRSBench Visual Grounding and VQA benchmark dataset."""

    def __init__(self, data_root: Optional[Union[str, Path]] = None):
        self.data_root = Path(data_root) if data_root else None
        self.samples: List[VRSBenchSample] = []

    def load(self, annotations_path: Union[str, Path]) -> int:
        import json
        p = Path(annotations_path)
        if not p.exists():
            return 0
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
            for idx, item in enumerate(data if isinstance(data, list) else data.get("samples", [])):
                self.samples.append(
                    VRSBenchSample(
                        sample_id=item.get("id", f"vrs_{idx}"),
                        image_path=item.get("image_path", ""),
                        grounding_queries=item.get("grounding", []),
                        vqa_pairs=item.get("vqa", []),
                        caption=item.get("caption"),
                        metadata=item.get("metadata", {}),
                    )
                )
        return len(self.samples)

    def __len__(self) -> int:
        return len(self.samples)
