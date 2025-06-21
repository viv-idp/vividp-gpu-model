import json
import os
import csv
import sys
sys.path.append("../vividp-api/app")  # Add module path

from model_loader import load_model, predict_log

# Load model and vectorizer from correct paths
model, vectorizer = load_model("../vividp-api/app/vividp_model.pt", "../vividp-api/app/vectorizer.pkl")


# Set up file paths
log_file = "sample_logs.csv"
anomaly_file = "anomaly_store.json"
checkpoint_file = "collector_checkpoint.txt"

# Get checkpoint if exists
start_index = 0
if os.path.exists(checkpoint_file):
    with open(checkpoint_file, "r") as f:
        start_index = int(f.read().strip())

# Load all logs
with open(log_file, "r") as f:
    reader = csv.reader(f)
    logs = [row[0] for row in reader]

# Skip already processed logs
logs = logs[start_index:]

# Process new logs
new_anomalies = 0
for i, log in enumerate(logs):
    prediction, confidence = predict_log(model, vectorizer, log)
    
    if prediction == "anomaly":
        with open(anomaly_file, "a") as f:
            f.write(json.dumps({"log": log, "confidence": confidence}) + "\n")
        new_anomalies += 1

# Update checkpoint
with open(checkpoint_file, "w") as f:
    f.write(str(start_index + len(logs)))

print(f"✅ Done! Processed {len(logs)} new logs, stored {new_anomalies} new anomalies.")
