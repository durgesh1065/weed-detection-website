#!/usr/bin/env python3
import argparse
import base64
import json
import os
import sys
from io import BytesIO


def emit(payload):
    print(json.dumps(payload), flush=True)


def to_data_url(annotated_bgr):
    from PIL import Image

    rgb_array = annotated_bgr[:, :, ::-1]
    image = Image.fromarray(rgb_array)
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def build_detection_payload(result):
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
        method = "yolo-detection"
    else:
        label = "no_detection"
        confidence = 0.0
        method = "yolo-detection"

    annotated = result.plot(labels=True, conf=True, boxes=True, probs=False)
    return {
        "method": method,
        "label": label,
        "confidence": confidence,
        "detections": detections[:10],
        "annotatedImageBase64": to_data_url(annotated),
        "count": len(detections),
    }


def predict(model, image_path, conf, imgsz, device):
    results = model.predict(
        source=image_path,
        conf=conf,
        imgsz=imgsz,
        save=False,
        verbose=False,
        device=device,
    )
    if not results:
        return {
            "method": "yolo-detection",
            "label": "no_detection",
            "confidence": 0.0,
            "detections": [],
            "annotatedImageBase64": None,
            "count": 0,
        }

    return build_detection_payload(results[0])


def parse_request(raw_line):
    payload = json.loads(raw_line)
    request_id = payload.get("id")
    image_path = payload.get("imagePath")
    if not request_id:
        raise ValueError("Request is missing id.")
    if not image_path:
        raise ValueError("Request is missing imagePath.")
    return str(request_id), os.path.abspath(image_path)


def main():
    parser = argparse.ArgumentParser(description="Persistent weed detection inference worker.")
    parser.add_argument("--model", required=True, help="Path to YOLO .pt model")
    parser.add_argument("--conf", type=float, default=0.05, help="Detection confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--device", default="cpu", help="Inference device (cpu or cuda id)")
    args = parser.parse_args()

    model_path = os.path.abspath(args.model)
    if not os.path.exists(model_path):
        emit({"type": "fatal", "ok": False, "error": f"Model file not found: {model_path}"})
        sys.exit(1)
    if os.path.getsize(model_path) == 0:
        emit({"type": "fatal", "ok": False, "error": f"Model file is empty: {model_path}"})
        sys.exit(1)

    try:
        from ultralytics import YOLO

        model = YOLO(model_path)
    except Exception as exc:
        emit({"type": "fatal", "ok": False, "error": f"Failed to load model: {str(exc)}"})
        sys.exit(1)

    emit(
        {
            "type": "ready",
            "ok": True,
            "modelPath": model_path,
            "conf": args.conf,
            "imgsz": args.imgsz,
            "device": args.device,
        }
    )

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue

        request_id = None
        try:
            request_id, image_path = parse_request(line)
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")

            prediction = predict(
                model=model,
                image_path=image_path,
                conf=args.conf,
                imgsz=args.imgsz,
                device=args.device,
            )
            emit({"id": request_id, "ok": True, "prediction": prediction})
        except Exception as exc:
            emit({"id": request_id, "ok": False, "error": str(exc)})


if __name__ == "__main__":
    main()
