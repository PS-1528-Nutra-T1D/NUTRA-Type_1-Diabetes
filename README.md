# 🩸 GlucoPredict — Bi-LSTM + Transformer Glucose Telemetry

A clinical-grade blood glucose trajectory forecasting system for **Type-1 Diabetes** patients. It combines a state-of-the-art **Bidirectional LSTM + Multi-Head Self-Attention Transformer** deep learning model with a precision **Swiss-grid web interface** for real-time 30/60/120-minute predictions.

---

## 🤖 Model Architecture

The core model is a **dual-branch Bi-LSTM + Transformer** (`BiLSTMTransformer`) trained on the **OhioT1DM** dataset of real-world CGM physiological logs sampled at 5-minute intervals.

### How it works

1. **Input:** A sliding window of the last **12 CGM readings** (60 minutes of history) with 4 features each:
   - `GlucoseCGM` — raw CGM reading (mg/dL)
   - `HR` — heart rate (bpm)
   - `time_sin`, `time_cos` — circadian time-of-day encoding

2. **Dual-Branch Processing:**
   - **Bi-LSTM branch:** Bidirectional LSTM captures both past→future and future→past temporal dependencies simultaneously
   - **Transformer branch:** Multi-head self-attention (4 heads, 2 layers) learns global sequence relationships

3. **Fusion:** Both branch outputs are concatenated (3×hidden) and passed through a 3-layer MLP → single glucose prediction

4. **Multi-step rollout:** For 30/60/120 min horizons, predictions are rolled autoregressively with **physiological momentum decay** (γ=0.82, k=0.038) to prevent runaway monotonic drift

### Performance

| Metric | Value |
|--------|-------|
| RMSE | ~12–17 mg/dL |
| MAE | ~8–12 mg/dL |
| Clarke EGA Zone A+B | >98% |
| Prediction Horizons | 30 / 60 / 120 min |

---

## 🖥️ Web Interface

The frontend is designed following the **Swiss International Medical Ledger** aesthetic — a deliberate break from generic AI dashboards.

### Design Decisions
- **Typography:** Fraunces (high-contrast editorial serif) + JetBrains Mono (technical monospace)
- **Layout:** Asymmetric CSS Grid ledger structure — no centered hero, no pill buttons, no card grids
- **Palette:** Raw off-white (#F6F5F0) + hard ink black (#111213) + clinical signal red/amber/spruce
- **Dark Mode (Night Ledger):** Full CSS variable re-theming via `[data-theme="dark"]`, toggled persistently via `localStorage`

### Features
- 📁 CSV upload (requires `GlucoseCGM` + `HR` columns, min 12 rows)
- 📊 Interactive Chart.js glucose forecast chart (re-themed on dark mode toggle)
- ⚠️ Diagnostic banners for hypo (<70 mg/dL) and hyper (>180 mg/dL) readings
- 📋 30 / 60 / 120 min tab switcher
- 📥 Export forecast as CSV or copy to clipboard
- 🔀 **Workflow tab** — visual step-by-step diagram of how the prediction pipeline works
- 🌙 **Dark Mode toggle** — single-click Night Ledger theme, persists across sessions

---

## 📁 Repository Structure

```
NUTRA-Type_1-Diabetes/
├── backend/
│   ├── app.py                        # FastAPI server (predict, upload endpoints)
│   ├── bilstm_transformer_5min.pth   # Trained Bi-LSTM+Transformer weights
│   ├── feature_scaler.pkl            # StandardScaler for input features
│   ├── target_scaler.pkl             # StandardScaler for glucose output
│   ├── spike_test.csv                # Test CSV: post-meal glucose spike
│   ├── drop_test.csv                 # Test CSV: hypoglycemia recovery
│   └── stable_test.csv               # Test CSV: euglycemic flat trace
├── frontend/
│   ├── index.html                    # Main app UI (Swiss design + dark mode)
│   ├── login.html                    # Login page
│   └── profile.html                  # Patient profile page
├── Bi_LSTM_Transformer/
│   ├── Bi_LSTM_Transformer.ipynb     # Full training notebook
│   ├── bilstm_transformer_5min.pth   # Model weights (copy)
│   ├── actual_vs_predicted.png       # Validation curve
│   └── training_loss.png             # Training loss curve
├── model_training_and_graphs/
│   └── lstmtransformer.py            # Standalone training script
└── README.md
```

---

## 🚀 Running Locally

### 1. Start the backend
```bash
cd backend
pip install fastapi uvicorn torch pandas scikit-learn
uvicorn app:app --reload --port 8000
```

### 2. Open the frontend
Open `frontend/index.html` directly in your browser (no build step needed).

> The frontend calls `http://localhost:8000/predict` — make sure the backend is running first.

### 3. Upload a test CSV
Your CSV must have at minimum:
```
GlucoseCGM,HR
120,72
118,74
...  (at least 12 rows)
```

---

## 📊 Test Scenarios

| File | Pattern | Expected Output |
|------|---------|----------------|
| `spike_test.csv` | Post-meal rise | Peaks ~218 mg/dL then dips to ~105 mg/dL |
| `drop_test.csv` | Hypoglycemia | Recovers from 42 → ~108 mg/dL, stabilises |
| `stable_test.csv` | Euglycemic | Stays flat ~105–127 mg/dL |
| Custom dip test | Low start + rebound | Rises from 60 → ~145 mg/dL then plateaus |

---

## 🔭 Architecture Diagram

```
CSV Upload (GlucoseCGM + HR)
        │
        ▼
  Feature Engineering
  (time_sin, time_cos, StandardScaler)
        │
        ▼
  Sliding Window [12 × 4]
        │
   ┌────┴────┐
   │         │
Bi-LSTM   Transformer
(bidirectional) (4-head MHA)
   │         │
   └────┬────┘
        │  Concatenate [3×hidden]
        ▼
   Fusion MLP → Scaled prediction
        │
        ▼
  Inverse Scale → mg/dL
        │
   ┌────┴────┐
   │         │
30 min    60 / 120 min
(direct)  (autoregressive rollout
           with momentum decay)
```

---

## 🎓 Academic Context

This project is a **Final Year Diploma Project** demonstrating the integration of:
- Physiological deep learning (Bi-LSTM + Transformer)
- Clinical safety validation (Clarke EGA)
- Production-ready REST API (FastAPI)
- Swiss-grid precision web interface
