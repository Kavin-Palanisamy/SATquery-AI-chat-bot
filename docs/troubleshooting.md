# SatQuery AI — Troubleshooting Guide (Phase 32)
**SIH Problem Statement**: SIH26167 · **Architecture Version**: `v3.0.0`

---

## 1. Port 8000 Already in Use (`[Errno 10048]`)
If the server reports `[Errno 10048] error while attempting to bind on address ('0.0.0.0', 8000)`:
- This indicates a SatQuery server instance is already running in the background.
- You can directly open `http://127.0.0.1:8000` in your browser, or kill the existing process:
  ```powershell
  Get-Process -Id (Get-NetTCPConnection -LocalPort 8000).OwningProcess | Stop-Process -Force
  ```

---

## 2. GeoTIFF Reading Warnings
If you see rasterio affine matrix deprecation warnings:
- These are non-fatal upstream warnings from rasterio/starlette; all geospatial coordinates, band data, and transforms load normally.

---

## 3. LoRA Adapter Not Found
If `train-status` reports `NOT LOADED`:
- Check that the directory specified in `SATQUERY_RS_ADAPTER_PATH` exists on disk and contains `adapter_config.json` and `adapter_model.bin` / `adapter_model.safetensors`.
