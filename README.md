# Weed Detection Website (React + Python FastAPI Inference)

This project provides:
- React frontend (`frontend/`) for image upload and prediction display
- Python FastAPI backend (`backend/app.py`) for upload API and inference
- YOLO model weights stored at `backend/weed_detection_model.pt`

## Current Model Status

Model file is stored in backend:
- `backend/weed_detection_model.pt`

Backend default model path resolves automatically to:
- `<repo-root>/backend/weed_detection_model.pt`

## 1) Backend Setup

```bash
cd backend
python -m pip install -r requirements.txt
```

Run backend:

```bash
uvicorn app:app --host 0.0.0.0 --port 5000
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
- `MODEL_PATH=<custom path to .pt model>`
- `INFERENCE_CONF=0.05`
- `INFERENCE_IMGSZ=512`
- `INFERENCE_DEVICE=cpu`
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

- Backend is Python-only (FastAPI), no Node.js backend required.
- Model is loaded once on startup for faster repeated inference.
- Prediction response includes `annotatedImageBase64` so frontend can render tagged bounding-box output.
- CORS is fully open (`*`) by default.

## Deploy (Render + Vercel)

Render backend:
- Root directory: `backend`
- Environment: `Python`
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`

Vercel frontend:
- Root directory: `frontend`
- Framework: `Vite`
- Env var: `VITE_API_BASE_URL=https://your-render-service.onrender.com`
