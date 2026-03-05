import cors from "cors";
import express from "express";
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import multer from "multer";
import { runPrediction } from "./services/predictor.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PORT = Number(process.env.PORT || 5000);
const DEFAULT_MODEL_PATH = path.normalize(
  "C:/Users/DURGE/Downloads/weed detection website/weed_detection_model.pt"
);
const MODEL_PATH = process.env.MODEL_PATH || DEFAULT_MODEL_PATH;
const MAX_UPLOAD_MB = Number(process.env.MAX_UPLOAD_MB || 100);
const MAX_UPLOAD_BYTES = Math.max(1, MAX_UPLOAD_MB) * 1024 * 1024;
const IMAGE_EXTENSIONS = new Set([
  ".jpg",
  ".jpeg",
  ".png",
  ".webp",
  ".bmp",
  ".tif",
  ".tiff",
  ".gif",
  ".heic",
  ".heif"
]);

const app = express();

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: MAX_UPLOAD_BYTES },
  fileFilter: (_req, file, cb) => {
    const extension = path.extname(file.originalname || "").toLowerCase();
    const hasImageMime = Boolean(file.mimetype && file.mimetype.startsWith("image/"));
    const hasKnownImageExtension = IMAGE_EXTENSIONS.has(extension);
    if (hasImageMime || hasKnownImageExtension) {
      cb(null, true);
      return;
    }
    cb(
      new Error(
        "Unsupported file type. Upload an image file (jpg, jpeg, png, webp, bmp, tif, tiff, gif, heic, heif)."
      )
    );
  }
});

app.use(
  cors({
    origin: process.env.CORS_ORIGIN ? process.env.CORS_ORIGIN.split(",") : true
  })
);
app.use(express.json());

app.get("/api/health", async (_req, res) => {
  try {
    const stats = await fs.stat(MODEL_PATH);
    res.json({
      status: "ok",
      model: {
        path: MODEL_PATH,
        exists: true,
        sizeBytes: stats.size,
        ready: stats.size > 0
      }
    });
  } catch {
    res.json({
      status: "ok",
      model: {
        path: MODEL_PATH,
        exists: false,
        sizeBytes: 0,
        ready: false
      }
    });
  }
});

app.post("/api/predict", upload.single("image"), async (req, res) => {
  if (!req.file) {
    res.status(400).json({ success: false, error: "Missing image file." });
    return;
  }

  try {
    const prediction = await runPrediction({
      imageBuffer: req.file.buffer,
      originalName: req.file.originalname || "upload.jpg",
      modelPath: MODEL_PATH
    });

    res.json({
      success: true,
      prediction
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Prediction failed.";
    res.status(500).json({
      success: false,
      error: message
    });
  }
});

app.use((error, _req, res, _next) => {
  if (error instanceof multer.MulterError && error.code === "LIMIT_FILE_SIZE") {
    res.status(400).json({
      success: false,
      error: `File too large. Max upload size is ${MAX_UPLOAD_MB} MB.`
    });
    return;
  }
  const message = error instanceof Error ? error.message : "Unexpected server error.";
  res.status(400).json({ success: false, error: message });
});

app.listen(PORT, () => {
  console.log(`Weed detection API listening on http://localhost:${PORT}`);
});
