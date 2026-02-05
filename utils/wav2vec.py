import torch
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2Model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

processor = Wav2Vec2Processor.from_pretrained(
    "facebook/wav2vec2-base"
)
model = Wav2Vec2Model.from_pretrained(
    "facebook/wav2vec2-base"
).to(DEVICE)

model.eval()

def extract_embedding(audio_16k):
    """
    audio_16k: numpy array (1D), 16kHz
    returns: 768-dim embedding
    """
    inputs = processor(
        audio_16k,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        outputs = model(
            inputs.input_values.to(DEVICE)
        )

    # Mean pooling over time
    embedding = outputs.last_hidden_state.mean(dim=1)
    return embedding.cpu().numpy().flatten()
