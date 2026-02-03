import base64
import uuid
import os
import tempfile
import soundfile as sf
import numpy as np

def save_base64_audio(base64_string):
    # Decode Base64 to bytes
    audio_bytes = base64.b64decode(base64_string)

    # Create safe temp directory (works on Windows/Mac/Linux)
    temp_dir = tempfile.gettempdir()

    wav_path = os.path.join(temp_dir, f"{uuid.uuid4()}.wav")

    # Save bytes directly as WAV (assumes input is WAV-compatible)
    with open(wav_path, "wb") as f:
        f.write(audio_bytes)

    return wav_path
