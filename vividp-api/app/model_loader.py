import torch
import torch.nn as nn
import joblib  # 🔹 Needed for loading the vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Load model and vectorizer
def load_model(model_path, vectorizer_path):
    # Define the model structure (must match the one used during training)
    model = nn.Sequential(
        nn.Linear(59, 128),
        nn.ReLU(),
        nn.Linear(128, 1),
        nn.Sigmoid()
    )
    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # Load the saved vectorizer
    vectorizer = joblib.load(vectorizer_path)

    return model, vectorizer

# Predict single log line
def predict_log(model, vectorizer, log_line):
    vector = vectorizer.transform([log_line]).toarray()
    x = torch.tensor(vector, dtype=torch.float32)

    with torch.no_grad():
        output = model(x)

    prediction = "anomaly" if output.item() > 0.5 else "normal"
    confidence = round(float(output.item()), 2)

    return prediction, confidence
