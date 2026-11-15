# AGENTS.md - PS_T1D Blood Glucose Prediction (v2)

## Project Overview
Type-1 Diabetes glucose prediction using **Bi-LSTM + Transformer** on OhioT1DM subset (`subdatabaseII.csv`). Trains 5-min step-ahead for autoregressive 120-min rollout.

## Repository Structure (Flat)
```
Bi_LSTM+Transformer/
├── colab_train_v2.ipynb          # Training pipeline (Colab GPU)
├── bilstm_transformer_5min.pth   # Trained checkpoint (weights + scalers + config)
├── test_predictions.csv          # Test set predictions (5-min horizon)
├── training_loss.png             # Training curve
├── actual_vs_predicted.png       # Actual vs predicted plot
├── subdatabaseII.csv             # Training data (CGM, HR, timestamp, PtID)
├── check_data.py                 # Data quality validation (optional)
├── problem statement.txt         # Project context (120-min target)
└── AGENTS.md                     # This file
```

## Model: Bi-LSTM + Transformer (v2)
- **Architecture**: Dual-branch Bi-LSTM (bidirectional, 1 layer) + Transformer Encoder (2 layers) → Fusion MLP
- **Features**: `GlucoseCGM`, `HR`, `time_sin`, `time_cos` (4 features)
- **Window**: 12 steps (60 min at 5-min intervals)
- **Target**: Next glucose reading (5-min ahead, `HORIZON=1`)
- **Config**: `HIDDEN=64`, `HEADS=4`, `LAYERS=2`, `DROPOUT=0.1`, `WD=1e-5`
- **Training**: AdamW, MSELoss, ReduceLROnPlateau, early stopping (patience=5), AMP
- **Splits**: Chronological per-patient (70/15/15), HR median from train only
- **Cleaning**: Drop patients >30% missing glucose, cap glucose at 400 mg/dL, min 500 clean rows
- **Checkpoint**: Contains `model_state_dict`, scalers (mean/scale), `hr_median`, feature list, config

## Developer Commands

### Train (Colab)
```bash
# 1. Upload colab_train_v2.ipynb + subdatabaseII.csv to Colab
# 2. Runtime → Change runtime type → GPU (T4)
# 3. Run all cells sequentially
```
Outputs: `bilstm_transformer_5min.pth`, `test_predictions.csv`, plots

### Verify Checkpoint
```bash
python -c "
import torch
ckpt = torch.load('bilstm_transformer_5min.pth', map_location='cpu')
print('Keys:', list(ckpt.keys()))
print('Window:', ckpt['window_size'])
print('Features:', ckpt['feature_columns'])
print('HR median:', ckpt['hr_median'])
"
```

### Data Quality Check (Optional)
```bash
python check_data.py
```

## Autoregressive 120-min Inference (Not Yet Implemented)
**Strategy**: Train on 5-min step → roll forward 24 steps at inference
1. Load checkpoint + scalers
2. Input: last 12 readings (60 min)
3. For 24 steps: predict → append → update time features → repeat
4. Output: predictions at 5, 10, 15... 120 min ahead
5. Evaluate: MAE/RMSE per horizon + Clarke Error Grid

## Critical Conventions

| Issue | Detail |
|-------|--------|
| **No leakage** | HR median computed from train split only; scalers fit on train only |
| **Chronological splits** | Per-patient 70/15/15 — never random shuffle |
| **Time features** | `time_sin`/`time_cos` use 1440-min period; must match at inference |
| **MAX_GAP_MIN=5** | Windows crossing >5min gaps rejected; adjust if sampling differs |
| **Checkpoint format** | `weights_only=True` compatible (PyTorch 2.6+) |
| **PtID grouping** | All splits by patient — never mix patients |

## Known Gaps / Next Steps
- [ ] Autoregressive inference script (`infer.py`)
- [ ] Multi-horizon evaluation (MAE at 5/15/30/60/90/120 min)
- [ ] Clarke Error Grid implementation
- [ ] Scheduled sampling during training (reduce exposure bias)
- [ ] Config file (YAML) instead of hardcoded constants
- [ ] Extract insulin/carb/meal features from OhioT1DM XML

## Environment
- Python 3.10+
- PyTorch 2.0+ (CUDA 12.x for GPU training)
- scikit-learn, pandas, numpy, matplotlib
- tqdm (Colab progress bars)