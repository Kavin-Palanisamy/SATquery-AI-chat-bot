# SatQuery AI — PEFT / LoRA Training Guide (Phase 32)
**SIH Problem Statement**: SIH26167 · **Architecture Version**: `v3.0.0`

---

## 1. Domain Adaptation Architecture

SatQuery AI uses Parameter-Efficient Fine-Tuning (PEFT) with LoRA to adapt large Vision-Language Models (e.g. `Qwen/Qwen2-VL-2B-Instruct`) to remote sensing imagery:
- **Base Model:** `Qwen/Qwen2-VL-2B-Instruct`
- **Adapter Type:** LoRA (`r=16`, `lora_alpha=32`, `target_modules=["q_proj", "v_proj"]`)
- **Dataset:** `BigEarthNet.txt` formatted into multi-turn vision-language conversation pairs.

---

## 2. Running Training

```powershell
# Run LoRA fine-tuning
python training/train_lora.py --config training/configs/qwen_rs_vlm_lora.yaml
```

Training parameters are configured in [`training/configs/qwen_rs_vlm_lora.yaml`](file:///c:/Users/kavin/Downloads/satquery-ai-main/satquery-ai-main/training/configs/qwen_rs_vlm_lora.yaml).

---

## 3. Loading the Adapted Checkpoint in Production

To load your trained adapter:
```powershell
$env:SATQUERY_RS_ADAPTER_PATH="outputs/checkpoints/qwen2_vl_lora_final"
python scripts/run_server.py
```
When this environment variable points to a valid LoRA directory, SatQuery AI automatically loads the adapter weights and reports:
`Provenance: ADAPTED | RS Adaptation: LOADED`.
