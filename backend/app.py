import os
import torch
import torch.nn as nn
import numpy as np
import joblib
import pandas as pd
import io
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration from notebook
WINDOW_SIZE = 12
FEATURES = ["GlucoseCGM", "HR"]
D_MODEL = 64
N_HEADS = 4
NUM_LAYERS = 2
DROPOUT = 0.1

# This dynamically finds the folder where app.py is located, 
# so it will work regardless of whether you move it to C:\college notes or elsewhere.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = BASE_DIR 



# Model Definitions
class TransformerFeatureExtractor(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, dropout):
        super().__init__()
        self.input_projection = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.input_projection(x)
        x = self.transformer(x)
        x = x[:, -1, :]
        return self.output_layer(x)

# Load Models and Scalers
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transformer = TransformerFeatureExtractor(len(FEATURES), D_MODEL, N_HEADS, NUM_LAYERS, DROPOUT).to(device)

# Look for weight files specifically, avoiding directories
possible_weights = ["model_weights.pth", "transformer_glucose_model.pth", "best_transformer_weights.pth"]
weights_path = None

for name in possible_weights:
    path = os.path.join(MODEL_DIR, name)
    if os.path.exists(path):
        weights_path = path
        break

if weights_path:
    try:
        # Standard load
        transformer.load_state_dict(torch.load(weights_path, map_location=device, weights_only=False))
        print(f"Successfully loaded weights from {weights_path}")
    except Exception as e:
        print(f"Standard load failed for {weights_path}: {e}. Trying dict load...")
        try:
            state_dict = torch.load(weights_path, map_location=device, weights_only=False)
            if isinstance(state_dict, dict):
                if 'model_state_dict' in state_dict:
                    transformer.load_state_dict(state_dict['model_state_dict'])
                else:
                    transformer.load_state_dict(state_dict)
                print(f"Successfully loaded weights from dict in {weights_path}")
            else:
                print(f"Loaded object from {weights_path} is not a dict.")
        except Exception as e2:
            print(f"Critical failure loading weights: {e2}")
else:
    print(f"CRITICAL ERROR: No valid weight FILE found in {MODEL_DIR}")
    print(f"Found files: {os.listdir(MODEL_DIR)}")

transformer.eval()




linear_model = joblib.load(os.path.join(MODEL_DIR, "linear_regression_model.pkl"))
feature_scaler = joblib.load(os.path.join(MODEL_DIR, "feature_scaler.pkl"))
target_scaler = joblib.load(os.path.join(MODEL_DIR, "target_scaler.pkl"))

def predict_glucose(input_data):
    data = np.array(input_data, dtype=np.float32)
    data_2d = data.reshape(-1, len(FEATURES))
    scaled_data = feature_scaler.transform(data_2d).reshape(1, WINDOW_SIZE, len(FEATURES))
    
    tensor_data = torch.tensor(scaled_data, dtype=torch.float32).to(device)
    with torch.no_grad():
        x = transformer.input_projection(tensor_data)
        x = transformer.transformer(x)
        features = x[:, -1, :].cpu().numpy()
    
    pred_scaled = linear_model.predict(features)
    pred = target_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()[0]
    return float(pred)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        if not all(col in df.columns for col in FEATURES):
            raise HTTPException(status_code=400, detail=f"CSV must contain columns: {FEATURES}")
        
        if len(df) < WINDOW_SIZE:
            raise HTTPException(status_code=400, detail=f"CSV must have at least {WINDOW_SIZE} rows")
        
        history = df[FEATURES].tail(WINDOW_SIZE).values.tolist()
        
        # Predictions for 30, 60, 120 mins with 5-min intervals
        # 30 mins = 6 steps, 60 mins = 12 steps, 120 mins = 24 steps
        # Assuming 1 step = 5 minutes
        
        def generate_forecast(start_history, steps):
            current_hist = list(start_history)
            forecast = []
            last_hr = start_history[-1][1]
            for _ in range(steps):
                pred = predict_glucose(current_hist)
                forecast.append(pred)
                current_hist.append([pred, last_hr])
                current_hist.pop(0)
            return forecast

        p30 = generate_forecast(history, 6)
        p60 = generate_forecast(history, 12)
        p120 = generate_forecast(history, 24)
        
        current_glucose = history[-1][0]
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

