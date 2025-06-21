from fastapi import FastAPI, Request
import torch
import joblib
from app.model_loader import load_model, predict_log

app = FastAPI()

model = load_model("app/vividp_model.pt")
vectorizer = joblib.load("app/vectorizer.pkl")

@app.get("/")
def home():
    return {"message": "🔥 vividp Inference API is Running. Use POST /predict to analyze logs."}

@app.post("/predict")
async def predict_log_route(request: Request):
    data = await request.json()
    log = data.get("log")
    if not log:
        return {"error": "Log input required"}
    
    vector = vectorizer.transform([log]).toarray()
    prediction, confidence = predict_log(model, vector)
    
    return {
        "log": log,
        "prediction": "anomaly" if prediction == 1 else "normal",
        "confidence": f"{confidence:.2f}"
    }
