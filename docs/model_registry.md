# SatQuery AI — Model & Tool Registry (Phase 32)
**SIH Problem Statement**: SIH26167 · **Architecture Version**: `v3.0.0`

---

## 1. Centralized Model Registry

Located at `src/satquery/models/registry.py`, the `ModelRegistry` manages model lifecycle, lazy loading, and provenance tracking:

```
Registered Remote Sensing Models:
  [VQA               ] Qwen2-VL-RS                      | Provenance: MOCK / ADAPTED | Status: READY
  [GROUNDING         ] RS-SpectralHeuristic-Grounder-v1 | Provenance: HEURISTIC      | Status: LOADED
  [SEGMENTATION      ] RS-SpectralHeuristic-Segmenter-v1 | Provenance: HEURISTIC      | Status: LOADED
  [DETECTION         ] RS-Morphological-Detector-v1     | Provenance: HEURISTIC      | Status: LOADED
  [CHANGE            ] RS-ClassicalSpectralChange-v1    | Provenance: REAL           | Status: LOADED
  [OPTICAL_SAR_FUSION] RS-CrossModalSynergy-v1          | Provenance: REAL           | Status: LOADED
```

---

## 2. Tool Registry

Located at `src/satquery/tools/registry.py`, the `ToolRegistry` registers specialist operational tools:

| Tool Name | Class | Tasks Handled | Modalities |
| :--- | :--- | :--- | :--- |
| `InputValidationTool` | `InputValidationTool` | `input_validation` | GeoTIFF, PNG, JPEG |
| `RemoteSensingVQATool` | `RemoteSensingVQATool` | `vqa`, `scene_description` | Optical, Multispectral |
| `CaptioningTool` | `CaptioningTool` | `captioning`, `scene_description` | Optical, Multispectral |
| `RemoteSensingGroundingTool` | `RemoteSensingGroundingTool` | `grounding`, `spatial_localization` | Optical, Multispectral |
| `SegmentationTool` | `SegmentationTool` | `segmentation`, `segmentation_area` | Optical, Multispectral |
| `DetectionTool` | `DetectionTool` | `detection`, `detection_count` | Optical, Multispectral |
| `RemoteSensingChangeTool` | `RemoteSensingChangeTool` | `bitemporal_change`, `change_analysis`| Bitemporal Optical |
| `OpticalSARFusionTool` | `OpticalSARFusionTool` | `optical_sar_fusion`, `crossmodal_fusion`| Optical + SAR |
