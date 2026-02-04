import base64
import uuid
import wave

def save_base64_audio(audio_base64: str) -> str:
    """
    Converts base64 string to WAV file and returns file path.
    """

    try:
        audio_bytes = base64.b64decode(audio_base64)
    except Exception as e:
        raise ValueError("Invalid Base64 audio data") from e

    file_path = f"/tmp/{uuid.uuid4()}.wav"

    with open(file_path, "wb") as f:
        f.write(audio_bytes)

    return file_path
