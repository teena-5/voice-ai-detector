from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
import time

from config import API_KEY
from utils import save_base64_audio
from feature_extractor import extract_features
from model import predict

app = FastAPI(title="AI Voice Detection API")

@app.get("/")
def home():
    return {"message": "AI Voice Detection API is running"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

# This MATCHES GUVI-HCL tester exactly
class RequestBody(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str

@app.post("/detect-voice")
def detect_voice(data: RequestBody, x_api_key: str = Header(None)):

    # ---- AUTH CHECK (HEADER BASED) ----
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    start = time.time()

    # Convert GUVI field to your internal format
    wav_path = save_base64_audio(data.audioBase64)
    features = extract_features(wav_path)
    label, confidence = predict(features)

    return {
        "classification": label,
        "confidence": round(confidence, 4),
        "language": data.language,
        "processing_time_ms": int((time.time() - start) * 1000)
    }
