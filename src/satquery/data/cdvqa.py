from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class CDVQASample(BaseModel):
    sample_id: str
    image_t1_path: str
    image_t2_path: str
    question: str
    answer: str
    change_type: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CDVQAAdapter:
    """Adapter for Change Detection Visual Question Answering (CDVQA) datasets."""

    def __init__(self, data_root: Optional[Union[str, Path]] = None):
        self.data_root = Path(data_root) if data_root else None
        self.samples: List[CDVQASample] = []

    def load(self, annotations_path: Union[str, Path]) -> int:
        import json
        p = Path(annotations_path)
        if not p.exists():
            return 0
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
            for idx, item in enumerate(data if isinstance(data, list) else data.get("samples", [])):
                self.samples.append(
                    CDVQASample(
                        sample_id=item.get("id", f"cdvqa_{idx}"),
                        image_t1_path=item.get("image_t1", ""),
                        image_t2_path=item.get("image_t2", ""),
                        question=item.get("question", ""),
                        answer=item.get("answer", ""),
                        change_type=item.get("change_type", "generic"),
                        metadata=item.get("metadata", {}),
                    )
                )
        return len(self.samples)

    def __len__(self) -> int:
        return len(self.samples)
