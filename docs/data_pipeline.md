# SatQuery AI — Data Pipeline & Adaptation Datasets (Phase 32)
**SIH Problem Statement**: SIH26167 · **Architecture Version**: `v3.0.0`

---

## 1. Supported Remote Sensing Datasets

SatQuery AI provides unified adapters in `src/satquery/data/`:

1. **`BigEarthNet.txt` (BigEarthNet-v1.0):**
   - 590,326 multi-spectral Sentinel-2 and dual-pol Sentinel-1 patches.
   - 19 CORINE Land Cover classes for multi-label land-cover classification and VQA instruction tuning.
2. **`RSVQA` (Remote Sensing Visual Question Answering):**
   - High-Resolution (HR) and Low-Resolution (LR) question-answer pairs over optical satellite imagery.
3. **`VRSBench`:**
   - Benchmark for visual reasoning, referring expression grounding, and scene captioning.
4. **`CDVQA` (Change Detection Visual Question Answering):**
   - Bi-temporal image pairs with conversational questions concerning urban expansion, disaster flooding, and vegetation change.

---

## 2. Dataset Ingestion Scripts
```powershell
# 1. Download dataset index or samples
python scripts/download_datasets.py --dataset bigearthnet

# 2. Prepare dataset into VQA conversation JSONL
python scripts/prepare_datasets.py --dataset bigearthnet --input-file data/BigEarthNet.txt
```
