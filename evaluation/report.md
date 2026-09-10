# SatQuery AI — Benchmark Evaluation Report
**SIH Problem Statement**: SIH26167  
**Architecture Version**: `v3.0.0`

---

## 1. Evaluation Truthfulness Declaration

In strict compliance with the **Non-Negotiable Truthfulness Rules** of SatQuery AI:
- **Zero Hallucinated Metrics**: No synthetic benchmark accuracies, IoUs, or F1 scores are claimed.
- **Explicit Provenance**: Every registered specialist model explicitly declares its current operational mode (`REAL`, `ADAPTED`, `HEURISTIC`, `MOCK`, `UNAVAILABLE`).
- **Benchmark Readiness**: Dedicated dataset evaluation drivers are implemented and verified for **RSVQA**, **VRSBench**, **CDVQA**, and **BigEarthNet**.

---

## 2. Benchmark Status Summary

| Task | Benchmark Dataset | Specialist Model | Provenance | Status | Measured Score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **VQA** | RSVQA / BigEarthNet | `Qwen2-VL-RS` | `MOCK` / `ADAPTED` | Ready (Evaluation Pipeline Wired) | *Requires downloaded benchmark dataset* |
| **Grounding** | VRSBench | `RS-SpectralHeuristic-Grounder-v1` | `HEURISTIC` | Ready (Evaluation Pipeline Wired) | *Requires downloaded benchmark dataset* |
| **Change Detection** | CDVQA / OSCD | `RS-ClassicalSpectralChange-v1` | `REAL` | Ready (Evaluation Pipeline Wired) | *Requires downloaded benchmark dataset* |
| **Optical-SAR Fusion**| Sentinel-1/2 Pairs | `RS-CrossModalSynergy-v1` | `REAL` | Ready (Evaluation Pipeline Wired) | *Requires downloaded benchmark dataset* |

---

## 3. How to Run Benchmark Evaluation
```powershell
# Run evaluation on RSVQA benchmark
python training/evaluate.py --dataset rsvqa

# Run evaluation on VRSBench grounding benchmark
python training/evaluate.py --dataset vrsbench

# Run evaluation on CDVQA bi-temporal change benchmark
python training/evaluate.py --dataset cdvqa
```
