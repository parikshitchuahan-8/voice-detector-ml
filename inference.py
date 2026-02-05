import torch

_model = None

def get_model():
    global _model
    if _model is None:
        _model = load_model()      # jo bhi tumhara model load hai
        _model.to("cpu")
        _model.eval()
    return _model

def predict_from_base64(audio_b64):
    model = get_model()
    with torch.no_grad():
        # inference logic
        return {"label": "AI", "confidence": 0.82}
