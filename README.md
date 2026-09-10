# SatQuery AI — Agentic Multimodal Remote Sensing Assistant
### Smart India Hackathon 2026 · Problem Statement ID: SIH26167
**Team:** Saverra · **Institution:** IIT Madras BS Degree Programme  
**Theme:** Space Technology · **Category:** Software

---

## 🛰️ 1. Project Overview

**SatQuery AI** is an interactive, agentic vision-language assistant for Earth Observation (EO) and remote sensing data. Instead of applying a single generic model, SatQuery AI automatically interprets natural language queries, validates input modalities, selects specialist remote-sensing tools, and returns evidence-grounded spatial and textual insights with transparent provenance.

```
                   User Query + Satellite Image(s)
                                 ↓
                 [ Agentic Query & Task Router ]
                                 ↓
     ┌──────────────────┬──────────────────┬──────────────────┐
     ↓                  ↓                  ↓                  ↓
[Tool 1: VQA]    [Tool 2: Grounding]  [Tool 3: Change]   [Tool 4: Opt+SAR]
Single GeoTIFF    Neon Bounding Box    Bi-Temporal Pair   Co-registered Pair
     └──────────────────┴──────────────────┴──────────────────┘
                                 ↓
       [ Interactive UI with Overlays & Observable Audit Logs ]
```

---

## ✨ 2. Core Functional Pillars

| Pillar | Capability | Operational Output |
| :--- | :--- | :--- |
| **1. Single-Image VQA** | Answers domain-specific questions on optical & multispectral imagery. | Natural language explanation with latency and confidence telemetry. |
| **2. Visual Grounding** | Localizes target entities (water bodies, agricultural parcels, urban built-up, runways). | Interactive glowing bounding boxes with class tags drawn directly on the canvas. |
| **3. Bi-Temporal Change Analysis** | Ingests before/after ($T_1, T_2$) image pairs for disaster, flood, and urban change. | CDVQA description + colorized change heatmap overlay. |
| **4. Optical + SAR Fusion** | Ingests co-registered optical (Sentinel-2) + SAR radar (Sentinel-1) pairs. | All-weather built-up and water body extraction penetrating cloud cover. |
| **5. Agentic Orchestration** | Automatically routes queries to the optimal specialist tool. | Observable step-by-step execution trace recorded in `outputs/executions.jsonl`. |

---

## 💻 3. Quickstart & Setup Guide

### Prerequisites:
- **Operating System:** Windows 10/11, macOS, or Ubuntu/Linux
- **Python:** Python 3.10 or 3.11 installed
- **Git:** Git CLI installed
- **Hardware:** Works on both CPU and GPU (NVIDIA GPU with CUDA recommended for real VLM weights; CPU/Mock mode supported for edge/lightweight testing).

---

### Step 1: Clone Repository
```powershell
git clone https://github.com/Amityush-lgtm/satquery-ai.git
cd satquery-ai
```

---

### Step 2: Create and Activate Virtual Environment

**On Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
*(If you see an execution policy error on PowerShell, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first).*

**On macOS / Linux (Bash/Zsh):**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

### Step 3: Install Dependencies
```powershell
pip install --upgrade pip
pip install -r requirements.txt
pip install -e . --no-deps
```

---

### Step 4: Generate Multi-Modal Sample Datasets
```powershell
python scripts/create_sample_data.py
```
This generates:
- `data/samples/sample_patch.tif` (16-bit Multi-band GeoTIFF with CRS)
- `data/samples/sample_urban.png` (High-resolution optical PNG)
- `data/samples/sample_bitemporal_t1.png` & `sample_bitemporal_t2.png` (Flood change pair)
- `data/samples/sample_optical_s2.png` & `sample_sar_s1.png` (Cloudy optical + Penetrating SAR pair)

---

## 🚀 4. How to Run

### Option A: Launch Interactive Web Workspace (Recommended)

```powershell
# Windows PowerShell (Instant Fast Mode)
$env:SATQUERY_MOCK_MODEL="1"
python scripts/run_server.py

# Windows PowerShell (Production Qwen2-VL Model)
python scripts/run_server.py

# macOS / Linux Bash
python scripts/run_server.py
```

Open your browser:  
👉 **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

---

### Option B: Run Automated Test Suite
```powershell
pytest -v
```
All **30/30 unit & integration tests** will pass in under 2 seconds.

---

### Option C: Command-Line Interface (CLI)
```powershell
python -m satquery.vqa --image data/samples/sample_patch.tif --question "What is visible in this satellite image?" --mock
```

---

## 📂 5. Project Architecture

```
satquery-ai/
├── configs/
│   └── model.yaml                # Model matrix & hyperparameters
├── data/
│   └── samples/                  # Sample GeoTIFFs, PNGs, Bi-temporal & SAR pairs
├── notebooks/
│   └── 01_bigearthnet_exploration.ipynb  # BigEarthNet.txt exploration notebook
├── outputs/
│   ├── executions.jsonl          # Provenance audit logs
│   └── temp_uploads/             # Normalized raster previews
├── scripts/
│   ├── create_sample_data.py     # Sample dataset generator
│   └── run_server.py             # Hot-reloading server launcher
├── src/
│   └── satquery/
│       ├── agent/
│       │   └── router.py         # Agentic Task Router & Execution Tracer
│       ├── analysis/
│       │   ├── change.py         # Bi-temporal change detection & CDVQA
│       │   ├── crossmodal.py     # Optical + SAR cross-modal fusion
│       │   └── grounding.py      # Visual grounding & bounding box engine
│       ├── api/
│       │   └── main.py           # FastAPI backend (/health, /preview, /agent/analyze)
│       ├── geo/
│       │   └── image_loader.py   # Multi-band GeoTIFF loader & percentile stretch
│       ├── schemas/
│       │   └── vqa.py            # Pydantic schemas (AgentResponse, BoundingBox)
│       ├── utils/
│       │   └── logging.py        # Audit & provenance logger
│       └── vqa/
│           ├── inference.py      # VQA inference controller
│           └── model.py          # Model singleton (Qwen2-VL & MockVQAModel)
├── tests/                        # 30 automated pytest tests
├── web/
│   ├── index.html                # Frontend UI with workflow mode tabs
│   ├── style.css                 # Celestial dark-mode styling & canvas overlays
│   ├── app.js                    # Dynamic preview, bounding box renderer, trace drawer
│   └── three.min.js              # Three.js celestial background
├── pytest.ini                    # Pytest configuration
├── pyproject.toml                # Package configuration
├── requirements.txt              # Project dependencies
└── README.md
```

---

## 👥 Team Details (Team Saverra — IIT Madras BS Degree Programme)
- **Kumar Amityush** (Team Leader) — *AI/ML & Overall Development*
- **Shreya Shrikant Jadhav** — *Research, Testing & Evaluation*
- **Rashes Kumar Tripathy** — *Vision Language Models & Fine-tuning*
- **Samadrita Dutta Gupta** — *Backend & Integration*
- **Shivam Kumar** — *Frontend & Visualisation*
- **Pavitra Patel** — *Remote Sensing & Image Processing*
