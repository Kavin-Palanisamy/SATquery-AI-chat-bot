import os
import sys
from pathlib import Path

# Set Vercel serverless environment flags
os.environ.setdefault("SATQUERY_MOCK_MODEL", "1")
os.environ.setdefault("SATQUERY_ALLOW_MODEL_DOWNLOADS", "0")
os.environ.setdefault("SATQUERY_ENV", "vercel")

# Ensure repository root and src directory are in Python path
root_dir = Path(__file__).resolve().parent.parent
src_dir = root_dir / "src"

if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from satquery.api.main import app

# Vercel's Python runtime exposes 'app' as the ASGI entrypoint
