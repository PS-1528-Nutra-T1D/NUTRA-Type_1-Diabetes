
import os
import copy
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ==================================================
# 1. SETTINGS
# ==================================================

CSV_PATH = "subdatabaseII.csv"

WINDOW_SIZE = 12
BATCH_SIZE = 256
EPOCHS = 20
LEARNING_RATE = 0.001

HIDDEN_SIZE = 64
NUM_HEADS = 4
NUM_TRANSFORMER_LAYERS = 2
DROPOUT = 0.1

# Maximum allowed interval between consecutive readings.
# Change this if your dataset has a different sampling interval.
MAX_GAP_MINUTES = 5

MODEL_PATH = "hybrid_lstm_transformer.pth"

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", DEVICE)


# ==================================================
# 2. LOAD AND PREPROCESS DATA
# ==================================================

def load_data():

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            "CSV file not found. Keep subdatabaseII.csv "
            "in the same folder as transformer.py."
        )

    df = pd.read_csv(CSV_PATH)

    required_columns = [
        "ts", "PtID", "GlucoseCGM", "HR"
    ]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(
                f"Missing column: {column}. "
                f"Available columns: {list(df.columns)}"
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
        subset=["ts", "PtID"]
    ).copy()

    # Treat non-positive glucose readings as missing.
    df.loc[
        df["GlucoseCGM"] <= 0,
        "GlucoseCGM"
    ] = np.nan

    df = df.sort_values(
        ["PtID", "ts"]
    ).reset_index(drop=True)

    # Time-of-day features.
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

    print("\n========== DATASET ==========")
    print("Total rows:", len(df))
    print("Number of patients:", df["PtID"].nunique())
    print(
        "Missing glucose values:",
        df["GlucoseCGM"].isna().sum()
    )
    print(
        "Missing heart-rate values:",
        df["HR"].isna().sum()
    )

    return df


# ==================================================
# 3. SPLIT DATA CHRONOLOGICALLY
# ==================================================

def split_data(df):

    splits = {
        "train": [],
        "val": [],
        "test": []
    }

    # Split each patient's timeline separately.
    for patient_id, patient_data in df.groupby(
        "PtID", sort=False
    ):

        patient_data = (
            patient_data.sort_values("ts")
            .reset_index(drop=True)
        )

        n = len(patient_data)

        if n < WINDOW_SIZE + 3:
            continue

        train_end = int(n * 0.70)
        val_end = int(n * 0.85)

        if train_end >= WINDOW_SIZE + 1:
            splits["train"].append(
                patient_data.iloc[:train_end].copy()
            )

        if val_end - train_end >= WINDOW_SIZE + 1:
            splits["val"].append(
                patient_data.iloc[
                    train_end:val_end
                ].copy()
            )

        if n - val_end >= WINDOW_SIZE + 1:
            splits["test"].append(
                patient_data.iloc[val_end:].copy()
            )

    if not splits["train"]:
        raise ValueError(
            "Not enough data to create training sequences."
        )

    # Impute heart rate using training data only.
    training_data = pd.concat(
        splits["train"], ignore_index=True
    )

    hr_median = training_data["HR"].median()

    if pd.isna(hr_median):
        hr_median = 0.0

    for split_name in splits:
        for i in range(len(splits[split_name])):

            group = splits[split_name][i].copy()

            group["HR"] = group["HR"].fillna(
                hr_median
            )

            splits[split_name][i] = group

    print("\n========== DATA SPLIT ==========")

    for name in splits:
        row_count = sum(
            len(group) for group in splits[name]
        )
        print(f"{name}: {row_count} rows")

    return splits, float(hr_median)


# ==================================================
# 4. CREATE SEQUENCES
# ==================================================

FEATURES = [
    "GlucoseCGM",
    "HR",
    "time_sin",
    "time_cos"
]


def create_sequences(
    groups,
    feature_scaler,
    target_scaler,
    fit_scalers=False
):

    X_list = []
    y_list = []

    # Fit scalers using training data only.
    if fit_scalers:

        training_data = pd.concat(
            groups, ignore_index=True
        )

        valid = training_data["GlucoseCGM"].notna()

        if valid.sum() == 0:
            raise ValueError(
                "No valid glucose values in training data."
            )

        feature_scaler.fit(
            training_data.loc[
                valid, FEATURES
            ].to_numpy(dtype=np.float32)
        )

        target_scaler.fit(
            training_data.loc[
                valid, ["GlucoseCGM"]
            ].to_numpy(dtype=np.float32)
        )

    max_gap = pd.Timedelta(
        minutes=MAX_GAP_MINUTES
    )

    for group in groups:

        group = (
            group.sort_values("ts")
            .reset_index(drop=True)
        )

        values = group[FEATURES].to_numpy(
            dtype=np.float32
        )

        glucose_valid = (
            group["GlucoseCGM"]
            .notna()
            .to_numpy()
        )

        timestamps = group["ts"].tolist()

        # The final row is the target.
        for end in range(
            WINDOW_SIZE, len(group)
        ):

            start = end - WINDOW_SIZE

            # Require observed glucose for all input rows
            # and the next-reading target.
            if not glucose_valid[
                start:end + 1
            ].all():
                continue

            times = timestamps[
                start:end + 1
            ]

            gaps = [
                times[i + 1] - times[i]
                for i in range(len(times) - 1)
            ]

            # Reject windows that cross large time gaps.
            if not all(
                pd.Timedelta(0) < gap <= max_gap
                for gap in gaps
            ):
                continue

            input_sequence = values[
                start:end
            ]

            target_glucose = values[
                end, 0
            ]

            X_scaled = feature_scaler.transform(
                input_sequence
            ).astype(np.float32)

            y_scaled = target_scaler.transform(
                np.array(
                    [[target_glucose]],
                    dtype=np.float32
                )
            ).astype(np.float32)

            X_list.append(X_scaled)
            y_list.append(y_scaled[0])

    if len(X_list) == 0:

        return (
            np.empty(
                (0, WINDOW_SIZE, len(FEATURES)),
                dtype=np.float32
            ),
            np.empty(
                (0, 1),
                dtype=np.float32
            )
        )

    return (
        np.stack(X_list),
        np.stack(y_list)
    )


# ==================================================
# 5. PYTORCH DATASET
# ==================================================

class GlucoseDataset(Dataset):

    def __init__(self, X, y):

        self.X = torch.tensor(
            X, dtype=torch.float32
        )

        self.y = torch.tensor(
            y, dtype=torch.float32
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):

        return self.X[index], self.y[index]


# ==================================================
# 6. TRANSFORMER POSITIONAL ENCODING
# ==================================================

class PositionalEncoding(nn.Module):

    def __init__(
        self,
        hidden_size,
        max_len=100
    ):

        super().__init__()

        position = torch.arange(
            max_len, dtype=torch.float32
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0, hidden_size, 2
            ).float()
            * (-np.log(10000.0) / hidden_size)
        )

        pe = torch.zeros(
            max_len, hidden_size
        )

        pe[:, 0::2] = torch.sin(
            position * div_term
        )

        pe[:, 1::2] = torch.cos(
            position
            * div_term[:pe[:, 1::2].shape[1]]
        )

        self.register_buffer(
            "pe", pe.unsqueeze(0)
        )

    def forward(self, x):

        return x + self.pe[:, :x.size(1), :]


# ==================================================
# 7. HYBRID LSTM + TRANSFORMER MODEL
# ==================================================

class HybridLSTMTransformer(nn.Module):

    def __init__(self):

        super().__init__()

        # Project the input features into hidden space.
        self.projection = nn.Linear(
            len(FEATURES), HIDDEN_SIZE
        )

        self.position = PositionalEncoding(
            HIDDEN_SIZE
        )

        # Branch 1: LSTM
        self.lstm = nn.LSTM(
            input_size=HIDDEN_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=1,
            batch_first=True
        )

        # Branch 2: Transformer encoder
        encoder_layer = (
            nn.TransformerEncoderLayer(
                d_model=HIDDEN_SIZE,
                nhead=NUM_HEADS,
                dim_feedforward=HIDDEN_SIZE * 4,
                dropout=DROPOUT,
                batch_first=True,
                activation="gelu"
            )
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=NUM_TRANSFORMER_LAYERS
        )

        # Combine features from both branches.
        self.fusion = nn.Sequential(
            nn.Linear(
                HIDDEN_SIZE * 2,
                HIDDEN_SIZE
            ),
            nn.ReLU(),
            nn.Dropout(DROPOUT),

            nn.Linear(
                HIDDEN_SIZE,
                HIDDEN_SIZE // 2
            ),
            nn.ReLU(),

            nn.Linear(
                HIDDEN_SIZE // 2,
                1
            )
        )

    def forward(self, x):

        # Input shape:
        # (batch_size, window_size, number_of_features)

        x = self.projection(x)
        x = self.position(x)

        # LSTM processes the time sequence.
        lstm_output, _ = self.lstm(x)

        # Transformer learns attention relationships.
        transformer_output = self.transformer(x)

        # Take the final time-step representation
        # from both branches.
        lstm_features = lstm_output[:, -1, :]

        transformer_features = (
            transformer_output[:, -1, :]
        )

        # Feature fusion: both models contribute.
        combined = torch.cat(
            [
                lstm_features,
                transformer_features
            ],
            dim=1
        )

        # Predict the next glucose reading.
        prediction = self.fusion(combined)

        return prediction


# ==================================================
# 8. TRAIN ONE EPOCH
# ==================================================

def train_epoch(
    model,
    loader,
    optimizer,
    loss_function
):

    model.train()

    total_loss = 0.0

    for X, y in loader:

        X = X.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad(
            set_to_none=True
        )

        predictions = model(X)

        loss = loss_function(
            predictions, y
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        total_loss += (
            loss.item() * len(X)
        )

    return total_loss / len(loader.dataset)


# ==================================================
# 9. EVALUATION
# ==================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    loss_function
):

    model.eval()

    total_loss = 0.0
    predictions_list = []
    actual_list = []

    for X, y in loader:

        X = X.to(DEVICE)
        y = y.to(DEVICE)

        predictions = model(X)

        loss = loss_function(
            predictions, y
        )

        total_loss += (
            loss.item() * len(X)
        )

        predictions_list.append(
            predictions.cpu().numpy()
        )

        actual_list.append(
            y.cpu().numpy()
        )

    predictions = np.concatenate(
        predictions_list
    )

    actual = np.concatenate(
        actual_list
    )

    average_loss = (
        total_loss / len(loader.dataset)
    )

    return average_loss, predictions, actual


# ==================================================
# 10. MAIN TRAINING PROGRAM
# ==================================================

def main():

    # Load data.
    df = load_data()

    # Split each patient timeline.
    splits, hr_median = split_data(df)

    # Standardize input and target features.
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    X_train, y_train = create_sequences(
        splits["train"],
        feature_scaler,
        target_scaler,
        fit_scalers=True
    )

    X_val, y_val = create_sequences(
        splits["val"],
        feature_scaler,
        target_scaler
    )

    X_test, y_test = create_sequences(
        splits["test"],
        feature_scaler,
        target_scaler
    )

    print("\n========== SEQUENCES ==========")
    print("Training sequences:", len(X_train))
    print("Validation sequences:", len(X_val))
    print("Testing sequences:", len(X_test))

    if (
        len(X_train) == 0
        or len(X_val) == 0
        or len(X_test) == 0
    ):

        raise ValueError(
            "No valid sequences in one or more splits. "
            "Check missing glucose values and timestamp "
            "intervals. Adjust MAX_GAP_MINUTES to match "
            "your dataset's actual sampling interval."
        )

    # Data loaders.
    train_loader = DataLoader(
        GlucoseDataset(X_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        GlucoseDataset(X_val, y_val),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        GlucoseDataset(X_test, y_test),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # Create the combined neural network.
    model = HybridLSTMTransformer().to(DEVICE)

    loss_function = nn.MSELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-5
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=2
        )
    )

    best_val_loss = float("inf")
    best_state = None

    patience = 5
    no_improvement = 0

    train_losses = []
    val_losses = []

    print(
        "\nStarting LSTM + Transformer training..."
    )

    # Training loop.
    for epoch in range(EPOCHS):

        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            loss_function
        )

        val_loss, _, _ = evaluate(
            model,
            val_loader,
            loss_function
        )

        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f}"
        )

        # Save the best validation model.
        if val_loss < best_val_loss:

            best_val_loss = val_loss

            best_state = copy.deepcopy(
                model.state_dict()
            )

            no_improvement = 0

            torch.save(
                {
                    "model_state_dict": best_state,
                    "window_size": WINDOW_SIZE,
                    "hidden_size": HIDDEN_SIZE,
                    "num_heads": NUM_HEADS,
                    "transformer_layers":
                        NUM_TRANSFORMER_LAYERS,
                    "feature_columns": FEATURES,
                    "hr_median": hr_median,
                    "feature_mean":
                        feature_scaler.mean_.tolist(),
                    "feature_scale":
                        feature_scaler.scale_.tolist(),
                    "target_mean":
                        target_scaler.mean_.tolist(),
                    "target_scale":
                        target_scaler.scale_.tolist()
                },
                MODEL_PATH
            )

        else:

            no_improvement += 1

        # Stop if validation performance does not improve.
        if no_improvement >= patience:

            print("Early stopping triggered.")
            break

    if best_state is None:
        raise RuntimeError(
            "Training did not produce a valid model."
        )

    # Load the best model.
    model.load_state_dict(best_state)

    # ==================================================
    # 11. TEST THE MODEL
    # ==================================================

    test_loss, pred_scaled, actual_scaled = evaluate(
        model,
        test_loader,
        loss_function
    )

    predictions = target_scaler.inverse_transform(
        pred_scaled
    ).ravel()

    actual = target_scaler.inverse_transform(
        actual_scaled
    ).ravel()

    mae = mean_absolute_error(
        actual, predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual, predictions
        )
    )

    print("\n========== TEST RESULTS ==========")

    print(
        f"Test MSE (scaled): {test_loss:.6f}"
    )

    print(
        f"MAE: {mae:.4f} dataset glucose units"
    )

    print(
        f"RMSE: {rmse:.4f} dataset glucose units"
    )

    print("Saved model:", MODEL_PATH)

    # ==================================================
    # 12. PLOT TRAINING AND VALIDATION LOSS
    # ==================================================

    plt.figure(figsize=(9, 5))

    plt.plot(
        train_losses,
        label="Training Loss"
    )

    plt.plot(
        val_losses,
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss (scaled target)")

    plt.title(
        "Hybrid LSTM + Transformer Training"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "training_loss.png",
        dpi=150
    )

    plt.show()

    # ==================================================
    # 13. PLOT ACTUAL VS PREDICTED GLUCOSE
    # ==================================================

    count = min(500, len(actual))

    plt.figure(figsize=(12, 5))

    plt.plot(
        actual[:count],
        label="Actual Glucose"
    )

    plt.plot(
        predictions[:count],
        label="Predicted Glucose"
    )

    plt.xlabel("Test Sample")

    plt.ylabel(
        "Glucose (dataset units)"
    )

    plt.title(
        "Actual vs Predicted Glucose"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "test_predictions.png",
        dpi=150
    )

    plt.show()

    # ==================================================
    # 14. SAVE TEST PREDICTIONS
    # ==================================================

    results = pd.DataFrame(
        {
            "actual_glucose": actual,
            "predicted_glucose": predictions,
            "absolute_error": np.abs(
                actual - predictions
            )
        }
    )

    results.to_csv(
        "test_predictions.csv",
        index=False
    )

    print("\n========== COMPLETED ==========")

    print("Model: hybrid_lstm_transformer.pth")
    print("Graph: training_loss.png")
    print("Graph: test_predictions.png")
    print("Results: test_predictions.csv")


if __name__ == "__main__":
    main()
