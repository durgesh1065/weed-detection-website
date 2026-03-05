#!/usr/bin/env python3
import argparse
import json
import os
import sys


def emit(payload):
    print(json.dumps(payload), flush=True)


def predict_with_ultralytics(model_path, image_path):
    from ultralytics import YOLO

    model = YOLO(model_path)
    results = model.predict(source=image_path, save=False, verbose=False, device="cpu")
    if not results:
        return {
            "method": "yolo",
            "label": "no_detection",
            "confidence": 0.0,
            "detections": [],
        }

    result = results[0]
    names = getattr(result, "names", {}) or {}
    detections = []

    if getattr(result, "boxes", None) is not None and len(result.boxes) > 0:
        confs = result.boxes.conf.cpu().tolist()
        classes = [int(x) for x in result.boxes.cls.cpu().tolist()]
        for cls_idx, conf in zip(classes, confs):
            label = str(names.get(cls_idx, cls_idx)) if isinstance(names, dict) else str(cls_idx)
            detections.append({"label": label, "confidence": round(float(conf), 4)})
        detections.sort(key=lambda item: item["confidence"], reverse=True)
        top = detections[0]
        return {
            "method": "yolo-detection",
            "label": top["label"],
            "confidence": top["confidence"],
            "detections": detections[:5],
        }

    probs = getattr(result, "probs", None)
    if probs is not None:
        class_index = int(probs.top1)
        confidence = float(probs.top1conf.item())
        label = str(names.get(class_index, class_index)) if isinstance(names, dict) else str(class_index)
        return {
            "method": "yolo-classification",
            "label": label,
            "confidence": round(confidence, 4),
            "detections": [{"label": label, "confidence": round(confidence, 4)}],
        }

    return {
        "method": "yolo",
        "label": "no_detection",
        "confidence": 0.0,
        "detections": [],
    }


def predict_with_torch(model_path, image_path):
    import torch
    from PIL import Image
    from torchvision import transforms

    artifact = torch.load(model_path, map_location="cpu")
    class_names = None

    if hasattr(artifact, "eval"):
        model = artifact
        class_names = getattr(artifact, "class_names", None)
    elif isinstance(artifact, dict) and hasattr(artifact.get("model"), "eval"):
        model = artifact["model"]
        class_names = artifact.get("class_names") or artifact.get("labels")
    else:
        raise RuntimeError(
            "Unsupported PyTorch model artifact. Provide a serialized model object or update infer.py for your checkpoint."
        )

    model.eval()
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(tensor)
        if isinstance(output, (list, tuple)):
            output = output[0]
        if output.ndim > 2:
            output = output.reshape(output.shape[0], -1)
        logits = output.squeeze(0)
        probabilities = torch.softmax(logits, dim=0)
        confidence, index = torch.max(probabilities, dim=0)

    predicted_index = int(index.item())
    label = str(predicted_index)
    if isinstance(class_names, (list, tuple)) and 0 <= predicted_index < len(class_names):
        label = str(class_names[predicted_index])
    elif isinstance(class_names, dict):
        label = str(class_names.get(predicted_index, predicted_index))

    top_values, top_indices = torch.topk(probabilities, k=min(5, probabilities.shape[0]))
    detections = []
    for value, idx in zip(top_values.tolist(), top_indices.tolist()):
        item_label = str(idx)
        if isinstance(class_names, (list, tuple)) and 0 <= idx < len(class_names):
            item_label = str(class_names[idx])
        elif isinstance(class_names, dict):
            item_label = str(class_names.get(int(idx), idx))
        detections.append({"label": item_label, "confidence": round(float(value), 4)})

    return {
        "method": "torch-classification",
        "label": label,
        "confidence": round(float(confidence.item()), 4),
        "detections": detections,
    }


def main():
    parser = argparse.ArgumentParser(description="Run weed detection model inference.")
    parser.add_argument("--model", required=True, help="Path to model file (.pt)")
    parser.add_argument("--image", required=True, help="Path to image file")
    args = parser.parse_args()

    model_path = os.path.abspath(args.model)
    image_path = os.path.abspath(args.image)

    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if os.path.getsize(model_path) == 0:
            raise RuntimeError(
                f"Model file is empty: {model_path}. Replace it with your trained weights before prediction."
            )
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        yolo_error = None
        try:
            prediction = predict_with_ultralytics(model_path, image_path)
        except Exception as exc:
            yolo_error = str(exc)
            prediction = predict_with_torch(model_path, image_path)

        payload = {"ok": True, "prediction": prediction}
        if yolo_error:
            payload["fallback"] = "ultralytics->torch"
            payload["note"] = yolo_error
        emit(payload)
    except Exception as exc:
        emit({"ok": False, "error": str(exc), "type": exc.__class__.__name__})
        sys.exit(1)


if __name__ == "__main__":
    main()

