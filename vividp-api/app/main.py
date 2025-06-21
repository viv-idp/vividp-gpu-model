import json
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from app.model_loader import load_model

# ✅ Create FastAPI app
app = FastAPI()

# ✅ Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with frontend domain for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Load model and vectorizer
model, vectorizer = load_model("app/vividp_model.pt", "app/vectorizer.pkl")

# ✅ /predict endpoint
@app.post("/predict")
async def predict(payload: dict):
    log = payload.get("log")
    if not log:
        raise HTTPException(status_code=400, detail="Missing 'log' field in payload.")
    
    x = vectorizer.transform([log]).toarray()
    x_tensor = torch.tensor(x, dtype=torch.float32)
    with torch.no_grad():
        output = model(x_tensor)
    prediction = "anomaly" if output.item() > 0.5 else "normal"
    confidence = round(float(output.item()), 2)
    return {"log": log, "prediction": prediction, "confidence": confidence}

# ✅ /anomalies endpoint
@app.get("/anomalies")
async def get_anomalies():
    try:
        with open("app/anomaly_store.json", "r") as f:
            lines = f.readlines()
            return [json.loads(line) for line in lines]
    except FileNotFoundError:
        return []
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
