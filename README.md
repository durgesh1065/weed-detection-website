# Weed Detection Website (React + Node.js + Python Inference)

This project provides:
- React frontend (`frontend/`) for image upload and prediction display
- Node.js Express backend (`backend/`) for upload API and inference orchestration
- Python inference script (`backend/python/infer.py`) that loads a `.pt` model

## Current Model Status

Your root model file exists at:
- `weed_detection_model.pt`

At the time of setup, it is `0 bytes` (empty), so real prediction will fail until you replace it with your trained model weights.

Backend default model path is set directly to:
- `C:\Users\DURGE\Downloads\weed detection website\weed_detection_model.pt`

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
- `INFERENCE_IMGSZ=640`
- `INFERENCE_DEVICE=cpu`
- `INFERENCE_TIMEOUT_MS=45000`

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
