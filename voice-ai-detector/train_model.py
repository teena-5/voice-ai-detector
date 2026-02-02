import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import os

# Create training data (synthetic but valid)
X = []
y = []

# Human-like samples
for i in range(200):
    X.append(np.random.normal(loc=0.5, scale=0.1, size=17))
    y.append(0)

# AI-like samples
for i in range(200):
    X.append(np.random.normal(loc=0.8, scale=0.05, size=17))
    y.append(1)

# Train model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X, y)

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/voice_detector.pkl")

print("Model created successfully: models/voice_detector.pkl")
