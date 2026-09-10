import os
import sys
from pathlib import Path
import uvicorn
from dotenv import load_dotenv

# Ensure workspace root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "src"))

load_dotenv()

def main():
    # Render and cloud platforms inject the PORT environment variable.
    # Binding to 0.0.0.0 ensures the port is reachable both locally and on cloud servers.
    port_env = os.getenv("PORT") or os.getenv("SATQUERY_PORT") or "8000"
    port = int(port_env)
    host = os.getenv("SATQUERY_HOST", "0.0.0.0")
    reload = os.getenv("SATQUERY_RELOAD", "false").lower() in ("1", "true")

    print(f"Starting SatQuery AI Server on http://{host}:{port} ...")
    uvicorn.run("satquery.api.main:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    main()
