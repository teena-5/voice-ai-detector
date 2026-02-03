from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import time

from config import API_KEY
from utils import save_base64_audio
from feature_extractor import extract_features
from model import predict

# ---------- APP INIT ----------
app = FastAPI(title="AI Voice Detection API")

# ---------- HEALTH CHECKS ----------

@app.get("/")
def home():
    return {"message": "Voice AI Detector API is running"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

# ---------- REQUEST SCHEMA ----------

class RequestBody(BaseModel):
    audio_base64: str
    language: str
    api_key: str

# ---------- MAIN ENDPOINT ----------

@app.post("/detect-voice")
def detect_voice(data: RequestBody):

    # ---- AUTH CHECK (BODY-BASED) ----
    if data.api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    start = time.time()

    # ---- PROCESS AUDIO ----
    wav_path = save_base64_audio(data.audio_base64)
    features = extract_features(wav_path)
    label, confidence = predict(features)

    # ---- RESPONSE ----
    return {
        "classification": label,
        "confidence": round(confidence, 4),
        "language": data.language,
        "processing_time_ms": int((time.time() - start) * 1000)
    }
