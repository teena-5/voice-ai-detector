from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import time

from config import API_KEY
from utils import save_base64_audio
from feature_extractor import extract_features
from model import predict

app = FastAPI(title="AI Voice Detection API")

# ✅ HEALTH CHECK (for Render)
@app.get("/")
def home():
    return {"message": "Voice AI Detector API is running"}

# ✅ OPTIONAL but recommended
@app.get("/health")
def health_check():
    return {"status": "ok"}

# ✅ REQUEST BODY MODEL (VERY IMPORTANT)
class RequestBody(BaseModel):
    audio_base64: str
    language: str
    api_key: str

# ✅ THIS MUST EXIST (your main endpoint)
@app.post("/detect-voice")
def detect_voice(data: RequestBody):

    # API Key check
    if data.api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    start = time.time()

    wav_path = save_base64_audio(data.audio_base64)
    features = extract_features(wav_path)
    label, confidence = predict(features)

    explanation = {
        "pitch_variation": "Low" if label == "AI_GENERATED" else "Natural",
        "spectral_signature": "Over-smooth" if label == "AI_GENERATED" else "Organic",
        "voice_micro_fluctuations": "Missing" if label == "AI_GENERATED" else "Present",
        "model_reason": "AI voices show unnatural stability in pitch and rhythm"
    }

    return {
        "classification": label,
        "confidence": round(confidence, 4),
        "language": data.language,
        "explanation": explanation,
        "processing_time_ms": int((time.time() - start) * 1000)
    }
