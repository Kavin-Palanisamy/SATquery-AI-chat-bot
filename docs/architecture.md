# SatQuery AI — System Architecture (Phase 32)
**SIH Problem Statement**: SIH26167 · **Architecture Version**: `v3.0.0`

---

## 1. High-Level Architecture Overview

SatQuery AI is designed around a modular, multi-tier agentic architecture:

```
                            [ User Natural Language Query + Satellite Imagery ]
                                                     │
                                                     ▼
                                      [ Phase 3: Input Validation ]
                                      (GeoTIFF CRS / Bands / Modality)
                                                     │
                                                     ▼
                                      [ Phase 13: Query Intent Analyzer ]
                                                     │
                                                     ▼
                                      [ Phase 13: Execution Planner ]
                                        (Generates Plans A through F)
                                                     │
               ┌───────────────────────┬─────────────┴─────────────┬───────────────────────┐
               ▼                       ▼                           ▼                       ▼
      [ Plan A: VQA ]         [ Plan B: Segmentation ]    [ Plan C: Grounding ]   [ Plan D: Detection ]
      `RemoteSensingVLM`       `SegmentationEngine`        `GroundingEngine`       `DetectionEngine`
      (Qualitative Reasoning)  (Pure-Pixel Area Mask)      (Spatial Bounding Box)  (Connected Component)
               │                       │                           │                       │
               └───────────────────────┼───────────────────────────┴───────────────────────┘
                                       ▼
                       [ Plan E/F: Change / Opt+SAR Fusion ]
                        `ChangeEngine` & `FusionEngine`
                                       │
                                       ▼
                       [ Phase 17: Answer Synthesizer ]
                     (Zero Hallucination · Refusal on Deficit)
                                       │
                                       ▼
              [ Interactive Celestial Web UI · CLI · JSONL Audit Logs ]
```

---

## 2. Subsystem Breakdown

1. **Input Validation Layer (`src/satquery/geo/`):** Inspects single rasters and multi-temporal/cross-modal pairs, verifying CRS, channel dimensions, and physical sensor compatibility.
2. **Specialist Engine Layer (`src/satquery/analysis/`):** Houses pure algorithmic engines (`SegmentationEngine`, `DetectionEngine`, `GroundingEngine`, `ChangeEngine`, `FusionEngine`).
3. **Model & Tool Registries (`src/satquery/models/`, `src/satquery/tools/`):** Decoupled repositories registering available models and tools with explicit provenance metadata.
4. **Agent Orchestration Layer (`src/satquery/agent/`):** Manages intent analysis, declarative stage planning, tool execution, and evidence fusion.
5. **Presentation & API Layer (`src/satquery/api/`, `web/`):** Exposes FastAPI REST endpoints and the responsive celestial dark-mode web workspace.
