import librosa
import numpy as np
import base64
import io
import tempfile
import os

TARGET_SR = 16000

def load_audio(path):
    audio, sr = librosa.load(path, sr=TARGET_SR)
    return audio

def base64_to_audio(base64_str):
    audio_bytes = base64.b64decode(base64_str)

    # Write to temp file because librosa handles mp3 best via file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        audio, sr = librosa.load(tmp_path, sr=TARGET_SR)
    finally:
        os.remove(tmp_path)

    return audio
