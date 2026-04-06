# Voice Detector ML

Voice Detector ML is a lightweight FastAPI service for running audio classification inference from base64-encoded requests. The current service exposes a minimal API that loads a PyTorch model on demand and returns a predicted label with confidence.

## What It Does

- Exposes a health endpoint for service checks
- Accepts base64-encoded audio payloads
- Protects inference with an API key header
- Runs PyTorch inference on CPU
- Returns a label and confidence score

## Tech Stack

- Python
- FastAPI
- Uvicorn
- PyTorch
- NumPy

## API Endpoints

- `GET /` returns service status
- `POST /detect` runs audio classification

Example request headers:

- `x-api-key: <your-api-key>`

Example request body:

```json
{
  "audio": "<base64-audio-content>"
}
```

## Local Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set an API key:

```bash
set API_KEY=your-secret-key
```

4. Start the API:

```bash
uvicorn app:app --reload
```

## Notes

- The repository currently contains placeholder inference logic and should be paired with the actual model-loading implementation.
- For production use, keep the API key and model assets outside version control.
