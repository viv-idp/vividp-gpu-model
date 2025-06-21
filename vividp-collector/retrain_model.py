import json
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Load existing anomaly logs
with open("anomaly_store.json") as f:
    lines = f.readlines()

logs = [json.loads(l)["log"] for l in lines]
labels = [1] * len(logs)  # All are anomalies

# Add some fake normal logs for balance
normal_logs = [
    "Started container abc",
    "Liveness probe succeeded",
    "Pulling image registry/nginx:1.25",
    "Pod scheduled to node-1",
    "Kubelet initialized successfully"
] * 20
logs.extend(normal_logs)
labels.extend([0] * len(normal_logs))

# Vectorize
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(logs).toarray()
y = torch.tensor(labels, dtype=torch.float32).view(-1, 1)

# Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

# Torch
X_train = torch.tensor(X_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)

# Model
model = nn.Sequential(
    nn.Linear(X.shape[1], 128),
    nn.ReLU(),
    nn.Linear(128, 1),
    nn.Sigmoid()
)

loss_fn = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
for epoch in range(10):
    model.train()
    optimizer.zero_grad()
    y_pred = model(X_train)
    loss = loss_fn(y_pred, y_train)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_pred = model(X_val)
        val_loss = loss_fn(val_pred, y_val)
    print(f"Epoch {epoch+1}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

# Save
torch.save(model.state_dict(), "vividp_model.pt")
import joblib
joblib.dump(vectorizer, "vectorizer.pkl")

print("✅ Retrained model + vectorizer saved.")
