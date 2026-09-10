# SatQuery AI — Production Deployment Guide (Phase 32)
**SIH Problem Statement**: SIH26167 · **Architecture Version**: `v3.0.0`

---

## 1. Deployment Modes

1. **Deterministic CPU Demo Mode (Lightweight):**
   ```powershell
   $env:SATQUERY_MOCK_MODEL="1"
   python scripts/run_server.py
   ```
   Uses pure-pixel deterministic heuristic segmentation/grounding and mock VLM reasoning without requiring CUDA GPU or external model weights.

2. **Production GPU Mode (Qwen2-VL + LoRA):**
   ```powershell
   $env:SATQUERY_VQA_MODEL="Qwen/Qwen2-VL-2B-Instruct"
   $env:SATQUERY_RS_ADAPTER_PATH="outputs/checkpoints/qwen2_vl_lora_final"
   python scripts/run_server.py
   ```
   Loads real vision-language weights onto CUDA GPU with lazy loading.

---

## 2. API Server Execution
```powershell
uvicorn satquery.api.main:app --host 0.0.0.0 --port 8000 --reload
```
