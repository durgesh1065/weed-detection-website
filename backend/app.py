import base64
import io
import os
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(BASE_DIR / "weed_detection_model.pt"))).resolve()
INFERENCE_CONF = float(os.getenv("INFERENCE_CONF", "0.05"))
INFERENCE_IMGSZ = int(os.getenv("INFERENCE_IMGSZ", "512"))
INFERENCE_DEVICE = os.getenv("INFERENCE_DEVICE", "cpu")
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "100"))
MAX_UPLOAD_BYTES = max(1, MAX_UPLOAD_MB) * 1024 * 1024
ANNOTATED_MAX_SIDE = int(os.getenv("ANNOTATED_MAX_SIDE", "1280"))

ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
    ".gif",
    ".heic",
    ".heif",
}

app = FastAPI(title="Weed Detection API", version="3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)

_model = None
_model_error = None


def _encode_annotated_image(annotated_bgr: np.ndarray) -> str:
    rgb_array = annotated_bgr[:, :, ::-1]
    image = Image.fromarray(rgb_array)
    if ANNOTATED_MAX_SIDE > 0:
        image.thumbnail((ANNOTATED_MAX_SIDE, ANNOTATED_MAX_SIDE), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=76, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def _load_model() -> None:
    global _model, _model_error
    if not MODEL_PATH.exists():
        _model = None
        _model_error = f"Model file not found: {MODEL_PATH}"
        return
    if MODEL_PATH.stat().st_size == 0:
        _model = None
        _model_error = f"Model file is empty: {MODEL_PATH}"
        return
    try:
        _model = YOLO(str(MODEL_PATH))
        _model_error = None
    except Exception as exc:
        _model = None
        _model_error = f"Failed to load model: {exc}"


def _build_prediction_payload(result) -> dict:
    names = getattr(result, "names", {}) or {}
    detections = []
    boxes = getattr(result, "boxes", None)

    if boxes is not None and len(boxes) > 0:
        classes = [int(item) for item in boxes.cls.cpu().tolist()]
        confidences = boxes.conf.cpu().tolist()
        coordinates = boxes.xyxy.cpu().tolist()

        for cls_idx, confidence, bbox in zip(classes, confidences, coordinates):
            label = str(names.get(cls_idx, cls_idx)) if isinstance(names, dict) else str(cls_idx)
            detections.append(
                {
                    "label": label,
                    "confidence": round(float(confidence), 4),
                    "bbox": [round(float(value), 1) for value in bbox],
                }
            )

        detections.sort(key=lambda item: item["confidence"], reverse=True)
        top = detections[0]
        label = top["label"]
        confidence = top["confidence"]
    else:
        label = "no_detection"
        confidence = 0.0

    annotated = result.plot(labels=True, conf=True, boxes=True, probs=False)
    return {
        "method": "yolo-detection",
        "label": label,
        "confidence": confidence,
        "detections": detections[:10],
        "annotatedImageBase64": _encode_annotated_image(annotated),
        "count": len(detections),
    }


@app.on_event("startup")
def _startup() -> None:
    _load_model()


@app.exception_handler(HTTPException)
async def _handle_http_exception(_request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"success": False, "error": str(exc.detail)})


@app.exception_handler(Exception)
async def _handle_generic_exception(_request, exc: Exception):
    return JSONResponse(status_code=500, content={"success": False, "error": str(exc)})


@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Weed detection backend is running (Python/FastAPI).",
        "endpoints": ["/api/health", "/api/predict"],
    }


@app.get("/api/health")
def health():
    exists = MODEL_PATH.exists()
    size = MODEL_PATH.stat().st_size if exists else 0
    return {
        "status": "ok",
        "model": {
            "path": str(MODEL_PATH),
            "exists": exists,
            "sizeBytes": size,
            "ready": bool(exists and size > 0 and _model is not None),
        },
        "backend": "python-fastapi",
    }


@app.post("/api/predict")
async def predict(image: UploadFile | None = File(None)):
    if image is None:
        raise HTTPException(status_code=400, detail="Missing image file.")

    if _model is None:
        raise HTTPException(status_code=500, detail=_model_error or "Model is not ready.")

    extension = Path(image.filename or "").suffix.lower()
    has_image_mime = bool(image.content_type and image.content_type.startswith("image/"))
    has_known_extension = extension in ALLOWED_EXTENSIONS
    if not has_image_mime and not has_known_extension:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Upload an image file (jpg, jpeg, png, webp, bmp, tif, tiff, gif, heic, heif).",
        )

    content = await image.read()
    if not content:
        raise HTTPException(status_code=400, detail="Missing image file.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail=f"File too large. Max upload size is {MAX_UPLOAD_MB} MB.")

    try:
        pil_image = Image.open(io.BytesIO(content)).convert("RGB")
        np_image = np.array(pil_image)
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="Invalid image file.")

    # Ultralytics expects ndarray input in BGR order.
    np_image_bgr = np_image[:, :, ::-1]

    results = _model.predict(
        source=np_image_bgr,
        conf=INFERENCE_CONF,
        imgsz=INFERENCE_IMGSZ,
        save=False,
        verbose=False,
        device=INFERENCE_DEVICE,
    )
    if not results:
        prediction = {
            "method": "yolo-detection",
            "label": "no_detection",
            "confidence": 0.0,
            "detections": [],
            "annotatedImageBase64": None,
            "count": 0,
        }
    else:
        prediction = _build_prediction_payload(results[0])

    return {"success": True, "prediction": prediction}
