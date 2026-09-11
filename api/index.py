import os
import sys
import traceback
from pathlib import Path

# Set Vercel serverless environment flags
os.environ.setdefault("SATQUERY_MOCK_MODEL", "1")
os.environ.setdefault("SATQUERY_ALLOW_MODEL_DOWNLOADS", "0")
os.environ.setdefault("SATQUERY_ENV", "vercel")

# Resolve directories
current_file = Path(__file__).resolve()
api_dir = current_file.parent
task_root = api_dir.parent

# Register all possible search locations for the src directory
candidate_paths = [
    str(task_root / "src"),
    str(task_root),
    str(api_dir / "src"),
    str(api_dir),
    "/var/task/src",
    "/var/task",
]

for p in candidate_paths:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

try:
    from satquery.api.main import app
except Exception:
    # Diagnostic fallback: if any import fails, expose the exact traceback
    # instead of crashing with an opaque 500 FUNCTION_INVOCATION_FAILED.
    from fastapi import FastAPI
    from fastapi.responses import PlainTextResponse

    err_trace = traceback.format_exc()
    app = FastAPI(title="SatQuery AI - Diagnostic Mode")

    @app.api_route("/{path_name:path}", methods=["GET", "POST", "PUT", "DELETE"])
    async def vercel_diagnostic_handler(path_name: str):
        existing_paths = [p for p in candidate_paths if os.path.exists(p)]
        root_files = os.listdir(task_root) if task_root.exists() else ["task_root does not exist"]
        report = (
            "SatQuery AI — Vercel Startup Diagnostic\n"
            + "=" * 50 + "\n"
            + "Exception Traceback:\n"
            + err_trace + "\n"
            + "=" * 50 + "\n"
            + f"Current working directory: {os.getcwd()}\n"
            + f"__file__: {current_file}\n"
            + f"task_root: {task_root}\n"
            + f"Existing search paths: {existing_paths}\n"
            + f"Root directory files: {root_files}\n"
            + "sys.path:\n" + "\n".join(sys.path)
        )
        return PlainTextResponse(report, status_code=500)
