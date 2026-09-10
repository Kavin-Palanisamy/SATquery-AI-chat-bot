# SatQuery AI — Benchmark Evaluation (Phase 32)
**SIH Problem Statement**: SIH26167 · **Architecture Version**: `v3.0.0`

---

## 1. Benchmark Protocols

SatQuery AI includes dedicated task-appropriate benchmark metrics:
- **VQA:** Accuracy / Exact Match / Semantic Match on RSVQA / BigEarthNet QA pairs.
- **Visual Grounding:** Mean Intersection-over-Union (mIoU) and bounding box recall on VRSBench.
- **Bi-Temporal Change:** F1-score and Change Mask IoU on CDVQA.
- **Optical + SAR Fusion:** Cross-sensor accuracy under cloudy conditions.

---

## 2. Running Benchmark Evaluation

```powershell
# Run benchmark evaluation
python training/evaluate.py --dataset rsvqa
python training/evaluate.py --dataset vrsbench
python training/evaluate.py --dataset cdvqa
```

Results are saved to `evaluation/results.json` and summarized in `evaluation/report.md`.
