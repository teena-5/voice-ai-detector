import librosa
import numpy as np

def extract_features(wav_path):
    y, sr = librosa.load(wav_path, sr=16000)

    # ---- SAFETY: handle empty / silent audio ----
    if y is None or len(y) == 0 or np.all(y == 0):
        # Return exactly 17 neutral features (to match your model)
        return np.zeros(17)

    # If audio is extremely short, pad it
    if len(y) < 2048:
        y = np.pad(y, (0, 2048 - len(y)), mode="constant")

    # ---- NORMAL FEATURE EXTRACTION (17 features total) ----
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)  # 13 features

    zcr = np.mean(librosa.feature.zero_crossing_rate(y))                # 1 feature
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))  # 1
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))    # 1
    rms = np.mean(librosa.feature.rms(y=y))                                      # 1

    # Stack to make exactly 17 features
    features = np.hstack([
        mfcc,                         # 13
        np.array([zcr,                # +1 = 14
                   spectral_centroid, # +1 = 15
                   spectral_rolloff,  # +1 = 16
                   rms])             # +1 = 17
    ])

    return features
