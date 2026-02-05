from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import os

from inference import predict_from_base64

app = FastAPI()

API_KEY = os.getenv("API_KEY", "testkey123")


class AudioRequest(BaseModel):
    audio: str


@app.get("/")
def root():
    return {"status": "running"}


@app.post("/detect")
def detect_voice(
    data: AudioRequest,
    x_api_key: str = Header(None)
):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    return predict_from_base64(data.audio)
