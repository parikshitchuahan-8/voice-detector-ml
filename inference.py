import joblib
import numpy as np
from audio_utils import base64_to_audio
from utils.wav2vec import extract_embedding

clf = joblib.load("model/classifier.joblib")

def predict_from_base64(base64_audio):
    audio = base64_to_audio(base64_audio)
    emb = extract_embedding(audio).reshape(1, -1)

    prob = clf.predict_proba(emb)[0]
    pred = np.argmax(prob)

    label = "AI_GENERATED" if pred == 1 else "HUMAN"
    confidence = float(prob[pred])

    # Simple explainability
    reason = (
        "Unnaturally smooth spectral patterns detected"
        if label == "AI_GENERATED"
        else "Natural pitch and spectral variation detected"
    )

    return {
        "label": label,
        "confidence": round(confidence, 3),
        "reason": reason
    }
