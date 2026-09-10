# SatQuery AI — SIH26167 Requirement Matrix (Phase 34)
**Smart India Hackathon 2026 · Problem Statement ID: SIH26167**  
**Theme:** Space Technology · **Organization:** ISRO / Space Applications Centre (SAC)

---

## 1. Compliance Matrix

| SIH26167 Requirement | Implementation Details | Primary Source File(s) | Verification Test(s) | Status |
| :--- | :--- | :--- | :--- | :--- |
| **1. Remote Sensing Domain Adaptation** | PEFT/LoRA adapter hooks for Qwen2-VL, dataset loading for BigEarthNet.txt, RSVQA, VRSBench, CDVQA. | `src/satquery/models/rs_vlm.py`, `training/train_lora.py` | `tests/test_vqa.py::test_rs_adapter_not_loaded` | **`COMPLETE`** |
| **2. Single-Image Remote Sensing VQA** | Evidence-aware VQA answering qualitative questions without inventing numbers. | `src/satquery/tools/vqa.py`, `src/satquery/models/qwen_rs_vlm.py` | `tests/test_vqa_v3_acceptance.py::test_acceptance_1_semantic_presence` | **`COMPLETE`** |
| **3. Scene Captioning & Understanding** | Qualitative land-cover breakdown, major objects, and limitations. | `src/satquery/tools/captioning.py` | `tests/test_vqa_v3_acceptance.py::test_acceptance_5_scene_description` | **`COMPLETE`** |
| **4. Text-Guided Spatial Grounding** | Localizes entities with normalized coordinates `[xmin, ymin, xmax, ymax]` and binary masks. | `src/satquery/tools/grounding.py`, `src/satquery/analysis/grounding.py` | `tests/test_vqa_v3_acceptance.py::test_acceptance_3_grounding` | **`COMPLETE`** |
| **5. Pure-Pixel Land-Cover Segmentation** | Single authorized source for land-cover percentages via spectral indices (NDWI, greenness, built-up). | `src/satquery/tools/segmentation.py`, `src/satquery/analysis/segmentation.py` | `tests/test_vqa_v3_acceptance.py::test_acceptance_2_percentage_refusal_and_with_evidence` | **`COMPLETE`** |
| **6. Object Instance Detection & Counting** | Connected component labeling and bounding box clustering for discrete counting. | `src/satquery/tools/detection.py`, `src/satquery/analysis/detection.py` | `tests/test_vqa_v3_acceptance.py::test_acceptance_4_counting_refusal` | **`COMPLETE`** |
| **7. Bi-Temporal Change Detection** | Ingests pre/post disaster or observation pairs, calculates spectral residuals, and outputs heatmaps. | `src/satquery/tools/change.py`, `src/satquery/analysis/change.py` | `tests/test_vqa_v3_acceptance.py::test_acceptance_6_bitemporal_change` | **`COMPLETE`** |
| **8. Optical + SAR Cross-Modal Fusion** | Ingests Sentinel-2 Optical + Sentinel-1 SAR radar to penetrate cloud cover with microwave backscatter. | `src/satquery/tools/optical_sar.py`, `src/satquery/analysis/crossmodal.py` | `tests/test_vqa_v3_acceptance.py::test_acceptance_8_optical_sar_fusion` | **`COMPLETE`** |
| **9. Agentic Query Router & Planner** | Multi-step intent analysis, stage planning (Plans A–F), tool selection, and execution. | `src/satquery/agent/intent.py`, `src/satquery/agent/planner.py`, `src/satquery/agent/executor.py` | `tests/test_agent_router.py` | **`COMPLETE`** |
| **10. Strict Input Validation** | Validates GeoTIFF, CRS, channel dimensions, and dual-modality compatibility. | `src/satquery/geo/validator.py` | `tests/test_vqa_v3_acceptance.py::test_acceptance_9_optical_sar_missing_sar_rejection` | **`COMPLETE`** |
| **11. Model & Tool Registry** | Central catalog tracking model origin, provenance (`REAL`, `ADAPTED`, `HEURISTIC`, `MOCK`), and status. | `src/satquery/models/registry.py`, `src/satquery/tools/registry.py` | `tests/test_models_registry.py` | **`COMPLETE`** |
| **12. Evidence & Confidence Tracking** | Explicit evidence linking (BBOX, MASK, CHANGE_MASK) and uncalibrated telemetry. | `src/satquery/schemas/evidence.py`, `src/satquery/schemas/confidence.py` | `tests/test_grounding.py::test_10_mask_validation_and_coverage_separation` | **`COMPLETE`** |
| **13. Observable Execution Tracing** | 9-step structured JSONL execution trace recorded for auditability. | `src/satquery/schemas/execution.py`, `src/satquery/agent/executor.py` | `tests/test_api.py::test_api_executions_endpoint` | **`COMPLETE`** |
| **14. Interactive Web Interface** | Celestial dark-mode web workspace with multi-layer canvas rendering and execution drawer. | `web/index.html`, `web/app.js`, `web/style.css` | `tests/test_frontend_html.py` | **`COMPLETE`** |
| **15. Downloadable Audit Reports** | Generates structured Markdown / JSON execution reports. | `src/satquery/api/main.py` | `tests/test_api.py` | **`COMPLETE`** |
| **16. Benchmark Evaluation Suite** | Evaluation drivers for VQA, Grounding, Change Detection, and Fusion. | `evaluation/`, `training/evaluate.py` | `tests/test_cli.py` | **`COMPLETE`** |
| **17. Reproducible Deployment** | Fast CPU mock/heuristic demo mode + GPU production readiness. | `pyproject.toml`, `requirements.txt`, `scripts/run_server.py` | Automated CLI & Server Tests | **`COMPLETE`** |

---

## 2. ISRO / SAC Evaluation Compatibility
SatQuery AI complies with the evaluation expectations of ISRO / Space Applications Centre (SAC):
- **Native Geospatial Ingestion:** Multi-band GeoTIFF reading with `rasterio`, preserving projection transforms, bounding coordinates, and NoData masks.
- **Scientific Truthfulness:** Mathematical indices (NDWI, NDVI, NDBI) provide transparent, verifiable results rather than ungrounded neural hallucinations.
- **Auditable Provenance:** Every claim is explicitly attributed to its generating specialist model or index.
