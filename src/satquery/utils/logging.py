import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

# Structured log format
LOG_FORMAT = "%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(
    level=os.getenv("SATQUERY_LOG_LEVEL", "INFO"),
    format=LOG_FORMAT,
    handlers=[logging.StreamHandler(sys.stdout)],
)

def get_logger(name: str) -> logging.Logger:
    """Returns a logger instance with standardized formatting."""
    return logging.getLogger(name)

# Provenance Execution Recording
PROVENANCE_FILE = Path("outputs") / "executions.jsonl"

def record_execution(
    task: str,
    model_name: str,
    image_path: str,
    question: str,
    answer: str,
    execution_time_sec: float,
    metadata: Optional[Dict[str, Any]] = None,
    confidence: Optional[float] = None,
) -> Dict[str, Any]:
    """Records an inference execution trace for auditability and provenance."""
    PROVENANCE_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    record = {
        "task": task,
        "model": model_name,
        "input": str(image_path),
        "question": question,
        "output": answer,
        "confidence": confidence,
        "execution_time_sec": round(execution_time_sec, 4),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata or {},
    }

    try:
        with open(PROVENANCE_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        logger = get_logger("satquery.provenance")
        logger.warning(f"Failed to write execution provenance: {e}")

    return record
