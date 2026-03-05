import { spawn } from "node:child_process";
import crypto from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const PYTHON_BIN = process.env.PYTHON_BIN || "python";
const INFERENCE_TIMEOUT_MS = Number(process.env.INFERENCE_TIMEOUT_MS || 45000);
const INFERENCE_CONF = process.env.INFERENCE_CONF || "0.05";
const INFERENCE_IMGSZ = process.env.INFERENCE_IMGSZ || "512";
const INFERENCE_DEVICE = process.env.INFERENCE_DEVICE || "cpu";

let workerState = null;

function parseJsonLine(line) {
  try {
    return JSON.parse(line);
  } catch {
    return null;
  }
}

function rejectAllPending(state, error) {
  for (const [id, request] of state.pending.entries()) {
    clearTimeout(request.timer);
    request.reject(error);
    state.pending.delete(id);
  }
}

function ensureWorker(modelPath) {
  if (
    workerState &&
    workerState.modelPath === modelPath &&
    workerState.process.exitCode === null &&
    !workerState.process.killed
  ) {
    return workerState;
  }

  if (workerState) {
    rejectAllPending(workerState, new Error("Inference worker restarted."));
    workerState.process.kill("SIGTERM");
    workerState = null;
  }

  const scriptPath = path.resolve(__dirname, "..", "python", "worker.py");
  const processArgs = [
    scriptPath,
    "--model",
    modelPath,
    "--conf",
    String(INFERENCE_CONF),
    "--imgsz",
    String(INFERENCE_IMGSZ),
    "--device",
    INFERENCE_DEVICE
  ];
  const child = spawn(PYTHON_BIN, processArgs, {
    stdio: ["pipe", "pipe", "pipe"]
  });

  const state = {
    process: child,
    modelPath,
    nextId: 1,
    pending: new Map(),
    stdoutBuffer: ""
  };
  workerState = state;

  child.stdout.on("data", (chunk) => {
    state.stdoutBuffer += chunk.toString();

    let newlineIndex = state.stdoutBuffer.indexOf("\n");
    while (newlineIndex >= 0) {
      const line = state.stdoutBuffer.slice(0, newlineIndex).trim();
      state.stdoutBuffer = state.stdoutBuffer.slice(newlineIndex + 1);
      newlineIndex = state.stdoutBuffer.indexOf("\n");
      if (!line) {
        continue;
      }

      const message = parseJsonLine(line);
      if (!message) {
        continue;
      }
      if (message.type === "ready") {
        continue;
      }

      const id = String(message.id ?? "");
      const request = state.pending.get(id);
      if (!request) {
        continue;
      }
      state.pending.delete(id);
      clearTimeout(request.timer);

      if (message.ok) {
        request.resolve(message.prediction);
        continue;
      }
      request.reject(new Error(message.error || "Inference failed."));
    }
  });

  child.stderr.on("data", (chunk) => {
    const output = chunk.toString().trim();
    if (output) {
      console.error(`[python] ${output}`);
    }
  });

  child.on("error", (error) => {
    rejectAllPending(state, error);
    if (workerState === state) {
      workerState = null;
    }
  });

  child.on("close", (code) => {
    const error = new Error(`Inference worker exited with code ${String(code ?? "unknown")}.`);
    rejectAllPending(state, error);
    if (workerState === state) {
      workerState = null;
    }
  });

  return state;
}

function requestPrediction({ modelPath, imagePath, timeoutMs = INFERENCE_TIMEOUT_MS }) {
  const state = ensureWorker(modelPath);
  return new Promise((resolve, reject) => {
    const id = String(state.nextId++);
    const timer = setTimeout(() => {
      state.pending.delete(id);
      reject(new Error("Model inference timed out."));
    }, timeoutMs);

    state.pending.set(id, { resolve, reject, timer });

    try {
      state.process.stdin.write(`${JSON.stringify({ id, imagePath })}\n`);
    } catch (error) {
      clearTimeout(timer);
      state.pending.delete(id);
      reject(error instanceof Error ? error : new Error("Failed to send request to inference worker."));
    }
  });
}

export async function runPrediction({ imageBuffer, originalName, modelPath }) {
  const extension = path.extname(originalName || ".jpg").replace(/[^.a-zA-Z0-9]/g, "") || ".jpg";
  const tempDir = path.resolve(__dirname, "..", "tmp");
  const fileName = `${crypto.randomUUID()}${extension}`;
  const imagePath = path.join(tempDir, fileName);

  await fs.mkdir(tempDir, { recursive: true });
  await fs.writeFile(imagePath, imageBuffer);

  try {
    return await requestPrediction({
      modelPath,
      imagePath
    });
  } finally {
    await fs.unlink(imagePath).catch(() => {});
  }
}
