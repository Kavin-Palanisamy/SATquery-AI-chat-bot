import json
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
from pydantic import BaseModel, Field


class BigEarthNetSample(BaseModel):
    """Structured sample representation for BigEarthNet.txt dataset."""
    sample_id: str = Field(..., description="Unique patch identifier (e.g., S2A_MSIL2A_...)")
    optical_path: Optional[str] = Field(None, description="Path to Sentinel-2 optical multispectral GeoTIFF/PNG")
    sar_path: Optional[str] = Field(None, description="Path to Sentinel-1 SAR radar GeoTIFF/PNG")
    caption: Optional[str] = Field(None, description="Natural language scene description")
    vqa_pairs: List[Dict[str, str]] = Field(default_factory=list, description="List of {question, answer} dictionaries")
    referring_expressions: List[Dict[str, Any]] = Field(default_factory=list, description="List of {query, bbox, mask} spatial annotations")
    land_cover_classes: List[str] = Field(default_factory=list, description="CORINE / BigEarthNet 19-class labels")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Acquisition date, tile CRS, solar angles")


class BigEarthNetTxtAdapter:
    """
    Parser and validator for BigEarthNet.txt remote sensing dataset.
    Supports optical, SAR, and multimodal paired inputs with VQA and grounding annotations.
    """

    def __init__(self, data_root: Optional[Union[str, Path]] = None):
        self.data_root = Path(data_root) if data_root else None
        self.samples: List[BigEarthNetSample] = []

    def load_from_file(self, file_path: Union[str, Path]) -> int:
        """Loads and validates dataset samples from a JSONL or formatted text file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"BigEarthNet.txt dataset file not found: {path}")

        loaded = 0
        with open(path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, 1):
                clean_line = line.strip()
                if not clean_line or clean_line.startswith("#"):
                    continue
                try:
                    data = json.loads(clean_line)
                    sample = self._parse_and_validate_sample(data, line_idx)
                    self.samples.append(sample)
                    loaded += 1
                except Exception as e:
                    # Log or skip malformed line
                    continue

        return loaded

    def _parse_and_validate_sample(self, raw: Dict[str, Any], idx: int) -> BigEarthNetSample:
        sample_id = raw.get("sample_id", f"sample_{idx}")
        opt_path = raw.get("optical_path") or raw.get("image_path")
        sar_path = raw.get("sar_path")

        # Validate bounding boxes
        ref_exprs = []
        for re_item in raw.get("referring_expressions", []):
            bbox = re_item.get("bbox")
            if bbox and len(bbox) == 4:
                ymin, xmin, ymax, xmax = bbox
                if 0.0 <= ymin <= ymax <= 1.0 and 0.0 <= xmin <= xmax <= 1.0:
                    ref_exprs.append(re_item)

        return BigEarthNetSample(
            sample_id=sample_id,
            optical_path=opt_path,
            sar_path=sar_path,
            caption=raw.get("caption"),
            vqa_pairs=raw.get("vqa_pairs", []),
            referring_expressions=ref_exprs,
            land_cover_classes=raw.get("classes", raw.get("land_cover_classes", [])),
            metadata=raw.get("metadata", {}),
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[BigEarthNetSample]:
        return iter(self.samples)
