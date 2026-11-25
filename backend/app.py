import os
import io
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException, UploadFile, File, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="GlucoDex API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILENAME = "bilstm_transformer_5min.pth"
MODEL_PATH = os.path.join(BASE_DIR, MODEL_FILENAME)

EXPECTED_MODEL_SIZE = 790680

MODEL_DOWNLOAD_URL = os.getenv(
    "MODEL_DOWNLOAD_URL",
    f"https://github.com/PS-1528-Nutra-T1D/NUTRA-Type_1-Diabetes/raw/main/backend/{MODEL_FILENAME}"
)

WINDOW_SIZE = 12
FEATURES = ["GlucoseCGM", "HR", "time_sin", "time_cos"]
HIDDEN_SIZE = 64
NUM_HEADS = 4
NUM_TRANSFORMER_LAYERS = 2
DROPOUT = 0.1

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

supabase: Client = None
if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
        print("Supabase client initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize Supabase client: {e}")
else:
    print("Warning: SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY not set. Running without database.")

EXPECTED_MODEL_SIZE = 790680

def verify_model_file(path):
    try:
        if not os.path.exists(path):
            return False, "File does not exist"
        size = os.path.getsize(path)
        if size == 0:
            return False, "File is empty"
        if size != EXPECTED_MODEL_SIZE:
            return False, f"Size mismatch: got {size}, expected {EXPECTED_MODEL_SIZE}"
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if "model_state_dict" not in ckpt and "proj.weight" not in ckpt:
            return False, "Invalid checkpoint structure"
        if "feature_mean" not in ckpt:
            return False, "Missing scaler parameters in checkpoint"
        return True, "OK"
    except Exception as e:
        return False, f"Load error: {e}"

def download_weights_if_missing():
    needs_download = False
    if not os.path.exists(MODEL_PATH):
        needs_download = True
        print(f"Model file not found at {MODEL_PATH}.")
    else:
        valid, msg = verify_model_file(MODEL_PATH)
        if not valid:
            needs_download = True
            print(f"Existing model file invalid: {msg}. Re-downloading...")
    
    if needs_download:
        print(f"Downloading from {MODEL_DOWNLOAD_URL}...")
        try:
            response = requests.get(MODEL_DOWNLOAD_URL, stream=True, timeout=60)
            response.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print("Download complete. Verifying...")
            valid, msg = verify_model_file(MODEL_PATH)
            if not valid:
                os.remove(MODEL_PATH)
                raise RuntimeError(f"Downloaded file verification failed: {msg}")
            print("Model weights successfully downloaded and verified.")
        except Exception as e:
            print(f"Failed to download/verify weights: {e}")
            if os.path.exists(MODEL_PATH):
                os.remove(MODEL_PATH)

download_weights_if_missing()

class PositionalEncoding(nn.Module):
    def __init__(self, hidden_size, max_len=200):
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

class BiLSTMTransformer(nn.Module):
    def __init__(
        self,
        num_features=len(FEATURES),
        hidden_size=HIDDEN_SIZE,
        num_heads=NUM_HEADS,
        transformer_layers=NUM_TRANSFORMER_LAYERS,
        dropout=DROPOUT
    ):
        super().__init__()
        self.proj = nn.Linear(num_features, hidden_size)
        self.pos = PositionalEncoding(hidden_size, max_len=200)
        self.bilstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0
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
            nn.Linear(3 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x):
        x = self.proj(x)
        x = self.pos(x)
        lstm_output, _ = self.bilstm(x)
        transformer_output = self.transformer(x)
        lstm_features = lstm_output[:, -1, :]
        transformer_features = transformer_output[:, -1, :]
        combined = torch.cat([lstm_features, transformer_features], dim=1)
        return self.fusion(combined)

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
        self.position = PositionalEncoding(hidden_size, max_len=100)
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
            nn.Dropout(dropout),
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

device = torch.device("cpu")
model = None
model_name = "Hybrid Bi-LSTM + Transformer"
feature_mean = None
feature_scale = None
target_mean = None
target_scale = None
hr_median = 75.0

if os.path.exists(MODEL_PATH):
    try:
        ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        
        hidden_size = ckpt.get("hidden_size", HIDDEN_SIZE)
        num_heads = ckpt.get("num_heads", NUM_HEADS)
        transformer_layers = ckpt.get("transformer_layers", NUM_TRANSFORMER_LAYERS)
        hr_median = float(ckpt.get("hr_median", 75.0))
        
        feature_mean = np.array(ckpt.get("feature_mean", [149.33, 76.67, 0.0, 0.0]), dtype=np.float32)
        feature_scale = np.array(ckpt.get("feature_scale", [62.50, 14.94, 0.71, 0.70]), dtype=np.float32)
        target_mean = float(ckpt.get("target_mean", [149.33])[0])
        target_scale = float(ckpt.get("target_scale", [62.50])[0])

        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        is_bilstm = any("bilstm" in k for k in state_dict.keys())

        if is_bilstm:
            model = BiLSTMTransformer(
                num_features=len(FEATURES),
                hidden_size=hidden_size,
                num_heads=num_heads,
                transformer_layers=transformer_layers,
                dropout=0.0
            ).to(device)
            model_name = "Hybrid Bi-LSTM + Transformer"
        else:
            model = HybridLSTMTransformer(
                num_features=len(FEATURES),
                hidden_size=hidden_size,
                num_heads=num_heads,
                transformer_layers=transformer_layers,
                dropout=0.0
            ).to(device)
            model_name = "Hybrid LSTM + Transformer"

        model.load_state_dict(state_dict)
        model.eval()
        print(f"Successfully initialized and loaded {model_name} from {MODEL_PATH}.")
    except Exception as e:
        print(f"Error loading model weights: {e}")
else:
    print(f"Warning: {MODEL_PATH} not found.")

def scale_features(arr_2d):
    return (arr_2d - feature_mean) / feature_scale

def inverse_scale_target(val_scaled):
    return (val_scaled * target_scale) + target_mean

def predict_single_step(window_data):
    scaled = scale_features(window_data)
    tensor_input = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(tensor_input)
    val_scaled = out.item()
    return float(inverse_scale_target(val_scaled))

MODEL_METRICS = {
    "model": model_name,
    "r2_score": 0.9926,
    "rmse": 5.3932,
    "mae": 3.2652,
    "eval_dataset": "OhioT1DM (subdatabaseII)",
    "test_samples": 65077,
    "horizon": "5-min step-ahead (Rollout 120m)"
}

RISK_ENUM = [
    "critical_hypo", "hypo", "stable", "hyper", 
    "critical_hyper", "rapid_drop", "rapid_rise"
]

def classify_risk(current_glucose: float, forecast_30: list, forecast_60: list, forecast_120: list, spike: float) -> str:
    min_30 = min(forecast_30)
    max_30 = max(forecast_30)
    min_60 = min(forecast_60)
    max_60 = max(forecast_60)
    min_120 = min(forecast_120)
    max_120 = max(forecast_120)
    
    if current_glucose < 54 or min_30 < 54 or min_60 < 54 or min_120 < 54:
        return "critical_hypo"
    if current_glucose < 70 or min_30 < 70 or min_60 < 70 or min_120 < 70:
        return "hypo"
    if current_glucose > 300 or max_30 > 300 or max_60 > 300 or max_120 > 300:
        return "critical_hyper"
    if current_glucose > 240 or max_30 > 240 or max_60 > 240 or max_120 > 240:
        return "hyper"
    if spike < -30 or (current_glucose < 90 and spike < -15):
        return "rapid_drop"
    if spike > 30 or (current_glucose > 180 and spike > 10):
        return "rapid_rise"
    return "stable"

def map_risk_to_chip_status(risk: str) -> str:
    if risk in ("critical_hypo", "hypo", "rapid_drop"):
        return "breach-low"
    if risk in ("critical_hyper", "hyper", "rapid_rise"):
        return "breach-high"
    return "safe"

def upsert_profile(clerk_user_id: str, email: str, full_name: str = None):
    if not supabase:
        return None
    try:
        data = {
            "clerk_user_id": clerk_user_id,
            "email": email,
        }
        if full_name:
            data["full_name"] = full_name
        result = supabase.table("profiles").upsert(data, on_conflict="clerk_user_id").execute()
        return result.data[0] if result.data else None
    except Exception as e:
        print(f"Failed to upsert profile: {e}")
        return None

async def get_current_user(
    x_user_id: str = Header(None, alias="X-User-Id"),
    x_user_email: str = Header(None, alias="X-User-Email"),
    x_user_name: str = Header(None, alias="X-User-Name")
):
    # Guest fallback: if headers missing or empty, use guest_demo_user
    user_id = x_user_id or "guest_demo_user"
    user_email = x_user_email or "guest@demo.local"
    user_name = x_user_name or "Guest User"
    return {"id": user_id, "email": user_email, "name": user_name}

@app.get("/")
def health_check():
    return {
        "status": "healthy",
        "model": model_name,
        "model_loaded": model is not None,
        "supabase_connected": supabase is not None,
        "metrics": MODEL_METRICS
    }

@app.get("/metrics")
def get_metrics():
    return MODEL_METRICS

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    user: dict = Depends(get_current_user)
):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded on server.")

    upsert_profile(user["id"], user["email"], user.get("name"))

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        if "GlucoseCGM" not in df.columns:
            raise HTTPException(status_code=400, detail="CSV must contain 'GlucoseCGM' column.")
            
        if "HR" not in df.columns:
            df["HR"] = hr_median
        else:
            df["HR"] = df["HR"].fillna(hr_median)

        if "ts" in df.columns:
            df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
            minutes = df["ts"].dt.hour * 60 + df["ts"].dt.minute
            minutes = minutes.fillna(0.0)
        else:
            minutes = pd.Series(np.zeros(len(df)))

        df["time_sin"] = np.sin(2 * np.pi * minutes / 1440).fillna(0.0)
        df["time_cos"] = np.cos(2 * np.pi * minutes / 1440).fillna(1.0)
        
        if len(df) < WINDOW_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"CSV must have at least {WINDOW_SIZE} rows. Provided: {len(df)}"
            )

        history_df = df[FEATURES].tail(WINDOW_SIZE).copy()
        history = history_df.to_numpy(dtype=np.float32)

        last_minute = float(minutes.iloc[-1]) if len(minutes) > 0 else 0.0

        def generate_forecast(start_history, steps, start_minutes=0.0):
            curr_hist = np.copy(start_history)
            forecast = []

            g_hist = curr_hist[:, 0]
            v1 = float(g_hist[-1] - g_hist[-2])
            v3 = float(g_hist[-1] - g_hist[-4]) / 3.0 if len(g_hist) >= 4 else v1
            v6 = float(g_hist[-1] - g_hist[-7]) / 6.0 if len(g_hist) >= 7 else v3

            vel = 0.5 * v1 + 0.3 * v3 + 0.2 * v6
            vel = float(np.clip(vel, -15.0, 15.0))

            curr_glucose = float(g_hist[-1])
            curr_min = float(start_minutes)
            last_hr = float(curr_hist[-1, 1])

            gamma = 0.88
            k_restore = 0.015
            target_eq = 115.0

            for _ in range(steps):
                curr_min = (curr_min + 5.0) % 1440.0
                step_sin = float(np.sin(2.0 * np.pi * curr_min / 1440.0))
                step_cos = float(np.cos(2.0 * np.pi * curr_min / 1440.0))

                nn_pred = predict_single_step(curr_hist)
                nn_delta = float(nn_pred - curr_glucose)

                vel = (vel * gamma) + (nn_delta * (1.0 - gamma)) - k_restore * (curr_glucose - target_eq)
                curr_glucose += vel

                curr_glucose = float(np.clip(curr_glucose, 40.0, 400.0))
                forecast.append(round(curr_glucose, 1))

                next_step = np.array([[curr_glucose, last_hr, step_sin, step_cos]], dtype=np.float32)
                curr_hist = np.vstack([curr_hist[1:], next_step])

            return forecast

        full_forecast = generate_forecast(history, 24, last_minute)
        p30 = full_forecast[:6]
        p60 = full_forecast[:12]
        p120 = full_forecast[:24]

        current_glucose = float(history[-1, 0])
        max_30 = max(p30)
        min_30 = min(p30)
        spike = (max_30 - current_glucose) if abs(max_30 - current_glucose) >= abs(min_30 - current_glucose) else (min_30 - current_glucose)

        risk_classification = classify_risk(current_glucose, p30, p60, p120, spike)
        chip_status = map_risk_to_chip_status(risk_classification)

        if current_glucose < 70 or min_30 < 70:
            feeling = "🚨 Critical Hypoglycemia Warning: Glucose is below clinical safety threshold (< 70 mg/dL). Immediate fast-acting carbohydrate intake needed."
        elif current_glucose > 240 or max_30 > 240:
            feeling = "⚠️ Severe Hyperglycemia Alert: Glucose is critically elevated (> 240 mg/dL). Check ketones and consider corrective bolus."
        elif spike > 30 or (current_glucose > 180 and spike > 10):
            feeling = "⚠️ High Glucose Spike: Rapid postprandial rise detected above clinical target (180 mg/dL). Keep an eye on peak trajectory."
        elif spike < -30 or (current_glucose < 90 and spike < -15):
            feeling = "🚨 Rapid Drop Alert: Glucose is declining rapidly. Prepare fast-acting glucose if descent continues."
        elif spike > 15:
            feeling = "📈 Moderate Rise: Glucose is increasing gradually. Monitor trend closely."
        elif spike < -15:
            feeling = "📉 Gradual Decline: Glucose is trending downward in a controlled manner."
        elif abs(spike) <= 15 and 70 <= current_glucose <= 180:
            feeling = "✅ Stable: Blood glucose is well-balanced within normal clinical target range."
        else:
            feeling = "Stable: Glucose trajectory is maintaining homeostatic balance."

        csv_batch_id = None
        if supabase:
            try:
                batch_data = {
                    "clerk_user_id": user["id"],
                    "filename": file.filename or "upload.csv",
                    "file_size_bytes": len(contents),
                    "row_count": len(df),
                    "columns_detected": list(df.columns),
                    "has_hr_column": "HR" in df.columns,
                    "has_ts_column": "ts" in df.columns,
                }
                batch_result = supabase.table("csv_upload_batches").insert(batch_data).execute()
                if batch_result.data:
                    csv_batch_id = batch_result.data[0]["id"]
            except Exception as e:
                print(f"Failed to log CSV batch: {e}")

        prediction_id = None
        if supabase:
            try:
                input_window = history.tolist()
                input_window_start = df["ts"].iloc[-WINDOW_SIZE].isoformat() if "ts" in df.columns and not pd.isna(df["ts"].iloc[-WINDOW_SIZE]) else datetime.now(timezone.utc).isoformat()
                input_window_end = df["ts"].iloc[-1].isoformat() if "ts" in df.columns and not pd.isna(df["ts"].iloc[-1]) else datetime.now(timezone.utc).isoformat()

                pred_data = {
                    "clerk_user_id": user["id"],
                    "csv_batch_id": csv_batch_id,
                    "model_name": model_name,
                    "model_version": os.path.basename(MODEL_PATH),
                    "input_window": input_window,
                    "input_window_start": input_window_start,
                    "input_window_end": input_window_end,
                    "current_glucose": current_glucose,
                    "forecast_30min": p30,
                    "forecast_60min": p60,
                    "forecast_120min": p120,
                    "risk_classification": risk_classification,
                    "feeling_text": feeling,
                    "metrics_snapshot": MODEL_METRICS,
                }
                pred_result = supabase.table("predictions").insert(pred_data).execute()
                if pred_result.data:
                    prediction_id = pred_result.data[0]["id"]
            except Exception as e:
                print(f"Failed to save prediction: {e}")

        if supabase and prediction_id:
            try:
                profile_result = supabase.table("profiles").select("target_glucose_low, target_glucose_high").eq("clerk_user_id", user["id"]).single().execute()
                if profile_result.data:
                    low_thresh = profile_result.data["target_glucose_low"]
                    high_thresh = profile_result.data["target_glucose_high"]
                    
                    alerts = []
                    
                    for horizon_min, forecast in [(30, p30), (60, p60), (120, p120)]:
                        min_val = min(forecast)
                        max_val = max(forecast)
                        
                        if min_val < low_thresh:
                            alerts.append({
                                "clerk_user_id": user["id"],
                                "prediction_id": prediction_id,
                                "alert_type": "hypo",
                                "horizon_minutes": horizon_min,
                                "predicted_value": min_val,
                                "threshold_value": low_thresh,
                                "breach_direction": "below",
                            })
                        if max_val > high_thresh:
                            alerts.append({
                                "clerk_user_id": user["id"],
                                "prediction_id": prediction_id,
                                "alert_type": "hyper",
                                "horizon_minutes": horizon_min,
                                "predicted_value": max_val,
                                "threshold_value": high_thresh,
                                "breach_direction": "above",
                            })
                    
                    if alerts:
                        supabase.table("alert_events").insert(alerts).execute()
            except Exception as e:
                print(f"Failed to save alert events: {e}")

        return {
            "predictions": {
                "30_min": p30,
                "60_min": p60,
                "120_min": p120
            },
            "feeling": feeling,
            "risk_classification": risk_classification,
            "chip_status": chip_status,
            "metrics": MODEL_METRICS
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/profile/{user_id}")
async def get_profile(user_id: str):
    if not supabase:
        raise HTTPException(status_code=503, detail="Database not configured")
    try:
        result = supabase.table("profiles").select("*").eq("clerk_user_id", user_id).single().execute()
        if not result.data:
            raise HTTPException(status_code=404, detail="Profile not found")
        return result.data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/profile")
async def upsert_profile_endpoint(
    profile_data: dict,
    user: dict = Depends(get_current_user)
):
    if not supabase:
        raise HTTPException(status_code=503, detail="Database not configured")
    try:
        allowed_fields = {
            "full_name", "age", "bio", "diabetes_type", "cgm_device",
            "target_glucose_low", "target_glucose_high", "preferred_unit", "avatar_url"
        }
        data = {"clerk_user_id": user["id"], "email": user["email"]}
        for key, value in profile_data.items():
            if key in allowed_fields:
                data[key] = value
        
        result = supabase.table("profiles").upsert(data, on_conflict="clerk_user_id").execute()
        if not result.data:
            raise HTTPException(status_code=500, detail="Failed to upsert profile")
        return result.data[0]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/preferences/{user_id}")
async def get_preferences(user_id: str):
    if not supabase:
        raise HTTPException(status_code=503, detail="Database not configured")
    try:
        result = supabase.table("user_preferences").select("*").eq("clerk_user_id", user_id).single().execute()
        if not result.data:
            return {
                "notify_high_glucose": True,
                "notify_low_glucose": True,
                "notify_spike": True,
                "notify_drop": True,
                "dark_mode_default": False,
                "audio_chime_enabled": True,
                "desktop_notify_enabled": False,
            }
        return result.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/preferences")
async def upsert_preferences_endpoint(
    prefs_data: dict,
    user: dict = Depends(get_current_user)
):
    if not supabase:
        raise HTTPException(status_code=503, detail="Database not configured")
    try:
        allowed_fields = {
            "notify_high_glucose", "notify_low_glucose", "notify_spike", "notify_drop",
            "dark_mode_default", "audio_chime_enabled", "desktop_notify_enabled"
        }
        data = {"clerk_user_id": user["id"]}
        for key, value in prefs_data.items():
            if key in allowed_fields:
                data[key] = value
        
        result = supabase.table("user_preferences").upsert(data, on_conflict="clerk_user_id").execute()
        if not result.data:
            raise HTTPException(status_code=500, detail="Failed to upsert preferences")
        return result.data[0]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))