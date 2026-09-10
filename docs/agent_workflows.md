# SatQuery AI — Agent Workflows & Execution Plans (Phase 32)
**SIH Problem Statement**: SIH26167 · **Architecture Version**: `v3.0.0`

---

## 1. Specialist Execution Plans

The SatQuery Agent uses declarative plans constructed by `ExecutionPlanner` in `src/satquery/agent/planner.py`:

### Plan A: `PLAN_A_VQA` (Semantic VQA & Captioning)
- **Trigger:** Queries asking for qualitative scene descriptions (*"Describe the land cover"*, *"Are there water bodies?"*).
- **Pipeline:** `InputValidation` $\rightarrow$ `QueryIntentAnalysis` $\rightarrow$ `SpecialistExecution` (`RemoteSensingVQATool`) $\rightarrow$ `AnswerSynthesis` $\rightarrow$ `AuditLogging`.
- **Anti-Hallucination Rule:** Refuses to produce fabricated percentage numbers without verified segmentation evidence.

### Plan B: `PLAN_B_SEGMENTATION_AREA` (Quantitative Land-Cover Area)
- **Trigger:** Inquiries requesting percentages or land-cover area (*"What percentage of the image is water?"*, *"How much vegetation?"*).
- **Pipeline:** `InputValidation` $\rightarrow$ `QueryIntentAnalysis` $\rightarrow$ `PixelSegmentation` (`SegmentationTool`) $\rightarrow$ `AreaMetricCalculation` $\rightarrow$ `EvidenceAwareVQASynthesis` $\rightarrow$ `AuditLogging`.
- **Source of Truth:** Pure-pixel calculation from `SegmentationEngine`.

### Plan C: `PLAN_C_GROUNDING` (Visual Grounding & Localization)
- **Trigger:** Localization queries (*"Find and highlight the water body"*, *"Where are the buildings?"*).
- **Pipeline:** `InputValidation` $\rightarrow$ `QueryIntentAnalysis` $\rightarrow$ `SpatialGrounding` (`RemoteSensingGroundingTool`) $\rightarrow$ `CoordinateNormalization` $\rightarrow$ `AuditLogging`.
- **Output:** Normalized bounding box coordinates `[xmin, ymin, xmax, ymax]` and binary mask. Rejects full-image boxes.

### Plan D: `PLAN_D_DETECTION_COUNT` (Discrete Instance Counting)
- **Trigger:** Inquiries asking for entity counts (*"How many buildings are visible?"*, *"Count storage tanks"*).
- **Pipeline:** `InputValidation` $\rightarrow$ `QueryIntentAnalysis` $\rightarrow$ `ObjectDetection` (`DetectionTool`) $\rightarrow$ `InstanceCounting` $\rightarrow$ `EvidenceSynthesis` $\rightarrow$ `AuditLogging`.
- **Source of Truth:** Pure-NumPy 4-connectivity connected component labeling.

### Plan E: `PLAN_E_CHANGE` (Bi-Temporal Change Analysis)
- **Trigger:** Two observation dates uploaded + change query (*"What changed between these two dates?"*).
- **Pipeline:** `PairValidation` $\rightarrow$ `QueryIntentAnalysis` $\rightarrow$ `SpectralDifferenceCalculation` (`RemoteSensingChangeTool`) $\rightarrow$ `ChangeHeatmapGeneration` $\rightarrow$ `ChangeDescriptionSynthesis` $\rightarrow$ `AuditLogging`.
- **Output:** Pixel difference residual, change heatmap, and alteration metrics.

### Plan F: `PLAN_F_CROSSMODAL` (Optical + SAR Cross-Modal Fusion)
- **Trigger:** Co-registered Optical RGB + SAR Radar pair uploaded (*"Use optical and SAR together"*).
- **Pipeline:** `PairModalityValidation` $\rightarrow$ `QueryIntentAnalysis` $\rightarrow$ `CrossModalSynergy` (`OpticalSARFusionTool`) $\rightarrow$ `FeatureExtraction` $\rightarrow$ `AuditLogging`.
- **Output:** Joint optical spectral and microwave backscatter feature extraction. Rejects single-image fake fusion.
