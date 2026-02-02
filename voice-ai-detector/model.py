import joblib

model = joblib.load("models/voice_detector.pkl")

def predict(features):
    prob = model.predict_proba([features])[0]
    label = "AI_GENERATED" if prob[1] > 0.5 else "HUMAN"
    confidence = float(max(prob))
    return label, confidence
