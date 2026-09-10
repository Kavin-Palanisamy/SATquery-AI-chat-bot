from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union
from pydantic import BaseModel, Field


class RSVQAQuestion(BaseModel):
    question_id: str
    image_path: str
    question: str
    answer: str
    question_type: str  # presence, count, comparison, area
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RSVQAAdapter:
    """Adapter for RSVQA-LR and RSVQA-HR benchmark datasets."""

    def __init__(self, data_dir: Optional[Union[str, Path]] = None):
        self.data_dir = Path(data_dir) if data_dir else None
        self.questions: List[RSVQAQuestion] = []

    def load(self, questions_file: Union[str, Path]) -> int:
        import json
        p = Path(questions_file)
        if not p.exists():
            return 0
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
            for item in data.get("questions", data if isinstance(data, list) else []):
                self.questions.append(
                    RSVQAQuestion(
                        question_id=str(item.get("id", len(self.questions))),
                        image_path=item.get("image", item.get("image_path", "")),
                        question=item.get("question", ""),
                        answer=item.get("answer", ""),
                        question_type=item.get("type", "presence"),
                        metadata=item.get("metadata", {}),
                    )
                )
        return len(self.questions)

    def __len__(self) -> int:
        return len(self.questions)
