
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# 1. SETTINGS
# ============================================================

CSV_PATH = "subdatabaseII.csv"
MODEL_PATH = "hybrid_lstm_transformer.pth"

WINDOW_SIZE = 12
HIDDEN_SIZE = 64
NUM_HEADS = 4
NUM_TRANSFORMER_LAYERS = 2
DROPOUT = 0.1

MAX_GAP_MINUTES = 5

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", DEVICE)


# ============================================================
# 2. CHECK FILES
# ============================================================

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(
        f"Dataset not found: {CSV_PATH}"
    )

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model checkpoint not found: {MODEL_PATH}"
    )


# ============================================================
# 3. LOAD SAVED CHECKPOINT
# ============================================================

# Only load checkpoints you trust.
try:
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )
except TypeError:
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

WINDOW_SIZE = checkpoint["window_size"]
HIDDEN_SIZE = checkpoint["hidden_size"]
NUM_HEADS = checkpoint["num_heads"]
NUM_TRANSFORMER_LAYERS = checkpoint["transformer_layers"]

FEATURES = checkpoint["feature_columns"]

feature_mean = np.array(
    checkpoint["feature_mean"], dtype=np.float32
)

feature_scale = np.array(
    checkpoint["feature_scale"], dtype=np.float32
)

target_mean = float(
    np.array(checkpoint["target_mean"]).reshape(-1)[0]
)

target_scale = float(
    np.array(checkpoint["target_scale"]).reshape(-1)[0]
)

hr_median = checkpoint["hr_median"]

print("Saved model checkpoint loaded.")
print("Window size:", WINDOW_SIZE)
print("Features:", FEATURES)


# ============================================================
# 4. POSITIONAL ENCODING
# ============================================================


class PositionalEncoding(nn.Module):

    def __init__(self, hidden_size, max_len=100):
        super().__init__()

        position = torch.arange(
            max_len, dtype=torch.float32
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0, hidden_size, 2, dtype=torch.float32
            ) * (-np.log(10000.0) / hidden_size)
        )

        pe = torch.zeros(1, max_len, hidden_size)

        pe[0, :, 0::2] = torch.sin(position * div_term)

        pe[0, :, 1::2] = torch.cos(
            position * div_term[:pe[0, :, 1::2].shape[-1]]
        )

        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class HybridLSTMTransformer(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        num_heads,
        transformer_layers,
        dropout=0.1
    ):
        super().__init__()

        self.projection = nn.Linear(
            input_size, hidden_size
        )

        self.position = PositionalEncoding(
            hidden_size
        )

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_layers
        )

        self.fusion = nn.Sequential(
    nn.Linear(hidden_size * 2, hidden_size),
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(hidden_size, 32),
    nn.ReLU(),
    nn.Linear(32, 1)
)

    def forward(self, x):

        x = self.projection(x)
        x = self.position(x)

        lstm_output, _ = self.lstm(x)

        transformer_output = self.transformer(x)

        lstm_last = lstm_output[:, -1, :]
        transformer_last = transformer_output[:, -1, :]

        combined = torch.cat(
            [lstm_last, transformer_last],
            dim=1
        )

        return self.fusion(combined)

# ============================================================
# 6. LOAD MODEL WEIGHTS
# ============================================================

model = HybridLSTMTransformer(
    input_size=len(FEATURES),
    hidden_size=HIDDEN_SIZE,
    num_heads=NUM_HEADS,
    transformer_layers=NUM_TRANSFORMER_LAYERS,
    dropout=DROPOUT
).to(DEVICE)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("Model weights loaded successfully.")


# ============================================================
# 7. LOAD AND PREPARE DATA
# ============================================================

df = pd.read_csv(CSV_PATH)

required_columns = [
    "ts", "PtID", "GlucoseCGM", "HR"
]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(
            f"Required column missing: {column}. "
            f"Found columns: {df.columns.tolist()}"
        )

df["ts"] = pd.to_datetime(
    df["ts"], errors="coerce"
)

df["GlucoseCGM"] = pd.to_numeric(
    df["GlucoseCGM"], errors="coerce"
)

df["HR"] = pd.to_numeric(
    df["HR"], errors="coerce"
)

df = df.dropna(
    subset=["ts", "PtID", "GlucoseCGM"]
).copy()

df = df[df["GlucoseCGM"] > 0].copy()

df = df.sort_values(
    ["PtID", "ts"]
).reset_index(drop=True)

# Time-of-day features must match training preprocessing.
minutes = (
    df["ts"].dt.hour * 60
    + df["ts"].dt.minute
)

df["time_sin"] = np.sin(
    2 * np.pi * minutes / 1440
)

df["time_cos"] = np.cos(
    2 * np.pi * minutes / 1440
)

df["HR"] = df["HR"].fillna(hr_median)

print("Dataset loaded:", df.shape)


# ============================================================
# 8. CREATE CHRONOLOGICAL TEST WINDOWS
# ============================================================

# This reproduces a 70% / 15% / 15% chronological split
# independently for each patient.
# Windows do not cross patient boundaries or data gaps.

test_windows = []
actual_values = []
timestamps = []
patient_ids = []

for patient_id, patient in df.groupby("PtID", sort=False):

    patient = patient.sort_values("ts").reset_index(
        drop=True
    )

    n = len(patient)

    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    # Test portion starts at val_end.
    for target_idx in range(
        max(val_end, WINDOW_SIZE), n
    ):

        start_idx = target_idx - WINDOW_SIZE

        # The whole input window and target must be in
        # the test section to avoid crossing split boundaries.
        if start_idx < val_end:
            continue

        window = patient.iloc[
            start_idx:target_idx
        ]

        target_row = patient.iloc[target_idx]

        # Check consecutive time intervals.
        times = pd.concat(
            [
                window["ts"],
                patient.loc[target_idx:target_idx, "ts"]
            ],
            ignore_index=True
        )

        gaps = times.diff().dropna().dt.total_seconds() / 60

        if not (
            (gaps > 0).all()
            and (gaps <= MAX_GAP_MINUTES).all()
        ):
            continue

        if window[FEATURES].isna().any().any():
            continue

        if target_row["GlucoseCGM"] <= 0:
            continue

        test_windows.append(
            window[FEATURES].to_numpy(
                dtype=np.float32
            )
        )

        actual_values.append(
            float(target_row["GlucoseCGM"])
        )

        timestamps.append(target_row["ts"])
        patient_ids.append(patient_id)


if len(test_windows) == 0:
    raise ValueError(
        "No valid test windows were created. "
        "Check your timestamp interval, feature columns, "
        "and the preprocessing used during training."
    )

X_test = np.asarray(
    test_windows, dtype=np.float32
)

actual = np.asarray(
    actual_values, dtype=np.float32
)

print("Number of test windows:", len(X_test))


# ============================================================
# 9. SCALE FEATURES USING SAVED SCALER VALUES
# ============================================================

# Avoid fitting a new scaler on test data.
feature_scale[feature_scale == 0] = 1.0

X_test_scaled = (
    X_test - feature_mean
) / feature_scale

X_test_tensor = torch.tensor(
    X_test_scaled,
    dtype=torch.float32,
    device=DEVICE
)


# ============================================================
# 10. PREDICT WITHOUT TRAINING
# ============================================================

predicted_scaled = []

with torch.no_grad():

    batch_size = 256

    for start in range(
        0, len(X_test_tensor), batch_size
    ):

        batch = X_test_tensor[
            start:start + batch_size
        ]

        outputs = model(batch)

        predicted_scaled.extend(
            outputs.detach().cpu().numpy().reshape(-1)
        )

predicted_scaled = np.asarray(
    predicted_scaled, dtype=np.float32
)

predicted = (
    predicted_scaled * target_scale
    + target_mean
)


# ============================================================
# 11. EVALUATION METRICS
# ============================================================

mae = mean_absolute_error(
    actual, predicted
)

rmse = np.sqrt(
    mean_squared_error(actual, predicted)
)

print("\n========== TEST RESULTS ==========")
print(f"MAE:  {mae:.3f}")
print(f"RMSE: {rmse:.3f}")
print("==================================")

# Note: These metrics are valid only if this script's
# model architecture, feature creation, windowing and split
# exactly match the original training script.


# ============================================================
# 12. SAVE ACTUAL AND PREDICTED VALUES
# ============================================================

results = pd.DataFrame({
    "Timestamp": timestamps,
    "PatientID": patient_ids,
    "Actual": actual,
    "Predicted": predicted
})

results.to_csv(
    "test_predictions.csv",
    index=False
)

print("Saved test_predictions.csv")


# ============================================================
# 13. PLOT ACTUAL VS PREDICTED
# ============================================================

plt.figure(figsize=(15, 7))

plt.plot(
    actual,
    label="Actual Glucose",
    linewidth=2
)

plt.plot(
    predicted,
    label="Predicted Glucose",
    linewidth=1.8,
    linestyle="--"
)

plt.title(
    "Actual vs Predicted Blood Glucose Levels\n"
    f"Hybrid LSTM + Transformer | MAE={mae:.2f}, "
    f"RMSE={rmse:.2f}"
)

plt.xlabel("Test Sample")
plt.ylabel("Blood Glucose Level")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    "actual_vs_predicted.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Saved actual_vs_predicted.png")


# ============================================================
# 14. OPTIONAL: PLOT A SMALLER SECTION
# ============================================================

number_to_plot = min(300, len(actual))

plt.figure(figsize=(15, 6))

plt.plot(
    actual[:number_to_plot],
    label="Actual Glucose",
    linewidth=2
)

plt.plot(
    predicted[:number_to_plot],
    label="Predicted Glucose",
    linestyle="--",
    linewidth=1.8
)

plt.title(
    f"Actual vs Predicted Glucose "
    f"(First {number_to_plot} Test Samples)"
)

plt.xlabel("Test Sample")
plt.ylabel("Blood Glucose Level")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    "actual_vs_predicted_first300.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Saved actual_vs_predicted_first300.png")
