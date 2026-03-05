# Weed Detection Website (React + Node.js + Python Inference)

This project provides:
- React frontend (`frontend/`) for image upload and prediction display
- Node.js Express backend (`backend/`) for upload API and inference orchestration
- Python worker (`backend/python/worker.py`) that loads a `.pt` model once and serves predictions

## Current Model Status

Model file is stored in backend:
- `backend/weed_detection_model.pt`

Backend default model path resolves automatically to:
- `<repo-root>/backend/weed_detection_model.pt`

## 1) Backend Setup

```bash
cd backend
npm install
```

Install Python dependencies:

```bash
cd python
python -m pip install -r requirements.txt
cd ..
```

Run backend:

```bash
npm run dev
```

Backend runs on `http://localhost:5000`.

## 2) Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

Frontend runs on `http://localhost:5173`.

## 3) Configure Environment (Optional)

Backend (`backend/.env` from `.env.example`):
- `PORT=5000`
- `PYTHON_BIN=python`
- `MODEL_PATH=<custom path to .pt model>`
- `INFERENCE_CONF=0.05`
- `INFERENCE_IMGSZ=512`
- `INFERENCE_DEVICE=cpu`
- `INFERENCE_TIMEOUT_MS=45000`
- `MAX_UPLOAD_MB=100`

Frontend (`frontend/.env` from `.env.example`):
- `VITE_API_BASE_URL=http://localhost:5000`

## 4) API Endpoints

- `GET /api/health`
  - Checks server and model readiness.
- `POST /api/predict`
  - Form-data field: `image`
  - Returns top prediction + confidence + detection list.

## Notes

- Backend now uses a persistent Python worker (`backend/python/worker.py`) so the model is loaded once and reused across requests for faster response time.
- Prediction response includes `annotatedImageBase64` so frontend can render tagged bounding-box output.
- CORS is fully open (`*`) by default.
