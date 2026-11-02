import os
import io
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="GlucoPredict AI API")

# Enable CORS for Netlify and local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths & Settings
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILENAME = "hybrid_lstm_transformer.pth"
MODEL_PATH = os.path.join(BASE_DIR, MODEL_FILENAME)

# Fallback download URL in case weights are stored on GitHub Releases / external host
MODEL_DOWNLOAD_URL = os.getenv(
    "MODEL_DOWNLOAD_URL",
    "https://github.com/PS-1528-Nutra-T1D/NUTRA-Type_1-Diabetes/raw/main/backend/hybrid_lstm_transformer.pth"
)

# Configuration defaults matching training
WINDOW_SIZE = 12
FEATURES = ["GlucoseCGM", "HR", "time_sin", "time_cos"]
HIDDEN_SIZE = 64
NUM_HEADS = 4
NUM_TRANSFORMER_LAYERS = 2
DROPOUT = 0.1

def download_weights_if_missing():
    """Ensure model file exists on disk. Downloads if missing."""
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) == 0:
        print(f"Model file not found at {MODEL_PATH}. Downloading from {MODEL_DOWNLOAD_URL}...")
        try:
            response = requests.get(MODEL_DOWNLOAD_URL, stream=True, timeout=60)
            response.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print("Model weights successfully downloaded.")
        except Exception as e:
            print(f"Failed to download weights: {e}")

# Trigger check/download on startup
download_weights_if_missing()

# ==================================================
# MODEL ARCHITECTURE
# ==================================================

class PositionalEncoding(nn.Module):
    def __init__(self, hidden_size, max_len=100):
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_size, 2).float()
            * (-np.log(10000.0) / hidden_size)
        )
        pe = torch.zeros(max_len, hidden_size)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]

class HybridLSTMTransformer(nn.Module):
    def __init__(
        self,
        num_features=len(FEATURES),
        hidden_size=HIDDEN_SIZE,
        num_heads=NUM_HEADS,
        transformer_layers=NUM_TRANSFORMER_LAYERS,
        dropout=DROPOUT
    ):
        super().__init__()
        self.projection = nn.Linear(num_features, hidden_size)
        self.position = PositionalEncoding(hidden_size)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_layers,
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x):
        x = self.projection(x)
        x = self.position(x)
        lstm_output, _ = self.lstm(x)
        transformer_output = self.transformer(x)
        lstm_features = lstm_output[:, -1, :]
        transformer_features = transformer_output[:, -1, :]
        combined = torch.cat([lstm_features, transformer_features], dim=1)
        return self.fusion(combined)

# ==================================================
# LOAD WEIGHTS & SCALER PARAMETERS
# ==================================================

device = torch.device("cpu")  # CPU inference is fast and avoids CUDA memory overhead on Render
model = None
feature_mean = None
feature_scale = None
target_mean = None
target_scale = None
hr_median = 75.0

if os.path.exists(MODEL_PATH):
    try:
        ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        
        # Load parameters if stored in checkpoint
        hidden_size = ckpt.get("hidden_size", HIDDEN_SIZE)
        num_heads = ckpt.get("num_heads", NUM_HEADS)
        transformer_layers = ckpt.get("transformer_layers", NUM_TRANSFORMER_LAYERS)
        hr_median = float(ckpt.get("hr_median", 75.0))
        
        feature_mean = np.array(ckpt.get("feature_mean", [143.82, 75.95, 0.0, 0.0]), dtype=np.float32)
        feature_scale = np.array(ckpt.get("feature_scale", [58.31, 14.23, 0.71, 0.70]), dtype=np.float32)
        target_mean = float(ckpt.get("target_mean", [143.82])[0])
        target_scale = float(ckpt.get("target_scale", [58.31])[0])

        model = HybridLSTMTransformer(
            num_features=len(FEATURES),
            hidden_size=hidden_size,
            num_heads=num_heads,
            transformer_layers=transformer_layers,
            dropout=0.0 # Inference mode
        ).to(device)
        
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)
        model.eval()
        print("Successfully initialized and loaded HybridLSTMTransformer.")
    except Exception as e:
        print(f"Error loading model weights: {e}")
else:
    print(f"Warning: {MODEL_PATH} not found.")

def scale_features(arr_2d):
    # Standard scale: (x - mean) / scale
    return (arr_2d - feature_mean) / feature_scale

def inverse_scale_target(val_scaled):
    # Inverse target: (scaled * scale) + mean
    return (val_scaled * target_scale) + target_mean

def predict_single_step(window_data):
    """
    window_data: np.ndarray of shape (WINDOW_SIZE, 4)
    Returns: unscaled glucose prediction as float
    """
    scaled = scale_features(window_data)
    tensor_input = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(tensor_input)
    val_scaled = out.item()
    return float(inverse_scale_target(val_scaled))

@app.get("/")
def health_check():
    return {
        "status": "healthy",
        "model": "Hybrid LSTM + Transformer",
        "model_loaded": model is not None
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded on server.")

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # Validate minimum columns: GlucoseCGM is mandatory
        if "GlucoseCGM" not in df.columns:
            raise HTTPException(status_code=400, detail="CSV must contain 'GlucoseCGM' column.")
            
        # Impute or default HR if missing
        if "HR" not in df.columns:
            df["HR"] = hr_median
        else:
            df["HR"] = df["HR"].fillna(hr_median)

        # Parse or create timestamps for time-of-day features
        if "ts" in df.columns:
            df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
            # If parsing succeeded, calculate minutes
            minutes = df["ts"].dt.hour * 60 + df["ts"].dt.minute
        else:
            minutes = pd.Series(np.zeros(len(df)))

        df["time_sin"] = np.sin(2 * np.pi * minutes / 1440).fillna(0.0)
        df["time_cos"] = np.cos(2 * np.pi * minutes / 1440).fillna(1.0)
        
        if len(df) < WINDOW_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"CSV must have at least {WINDOW_SIZE} rows. Provided: {len(df)}"
            )

        # Extract the last WINDOW_SIZE rows for inference
        history_df = df[FEATURES].tail(WINDOW_SIZE).copy()
        history = history_df.to_numpy(dtype=np.float32)

        # Auto-regressive multi-step forecast: 6 steps (30 min), 12 steps (60 min), 24 steps (120 min)
        def generate_forecast(start_history, steps):
            curr_hist = np.copy(start_history)
            forecast = []
            last_hr = curr_hist[-1, 1]
            last_sin = curr_hist[-1, 2]
            last_cos = curr_hist[-1, 3]

            for _ in range(steps):
                pred_glucose = predict_single_step(curr_hist)
                # Keep forecast non-negative
                pred_glucose = max(20.0, pred_glucose)
                forecast.append(round(pred_glucose, 1))

                # Shift window forward
                next_step = np.array([[pred_glucose, last_hr, last_sin, last_cos]], dtype=np.float32)
                curr_hist = np.vstack([curr_hist[1:], next_step])
            return forecast

        p30 = generate_forecast(history, 6)
        p60 = generate_forecast(history, 12)
        p120 = generate_forecast(history, 24)

        current_glucose = float(history[-1, 0])
        spike = p30[0] - current_glucose

        if spike > 40:
            feeling = "⚠️ High Glucose Spike: You may feel a burst of energy, but expect a 'sugar crash' (reactive hypoglycemia) soon, leading to sudden hunger and fatigue."
        elif spike > 20:
            feeling = "📈 Moderate Rise: Your glucose is increasing. You should feel stable, but keep an eye on the trend."
        elif spike < -40:
            feeling = "🚨 Rapid Drop: Warning! You may feel shaky, sweaty, dizzy, or confused. Immediate glucose intake may be needed."
        elif spike < -20:
            feeling = "📉 Gradual Decline: You might start feeling slightly hungry or less focused."
        elif abs(spike) <= 20:
            feeling = "✅ Stable: Your blood glucose is maintaining a healthy balance."
        else:
            feeling = "Neutral: No significant change detected."

        return {
            "predictions": {
                "30_min": p30,
                "60_min": p60,
                "120_min": p120
            },
            "feeling": feeling
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
