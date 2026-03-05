import { useEffect, useMemo, useRef, useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:5000";

function formatPercent(value) {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return "0.00%";
  }
  return `${(value * 100).toFixed(2)}%`;
}

function clampPercent(value) {
  if (!Number.isFinite(value)) {
    return 0;
  }
  return Math.max(0, Math.min(100, value));
}

export default function CApp() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [prediction, setPrediction] = useState(null);
  const [responseTimeMs, setResponseTimeMs] = useState(null);
  const [modelReady, setModelReady] = useState(false);
  const [modelStatusLoading, setModelStatusLoading] = useState(true);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [meterPercent, setMeterPercent] = useState(0);
  const meterPercentRef = useRef(0);

  useEffect(() => {
    let active = true;

    async function checkHealth() {
      try {
        const response = await fetch(`${API_BASE}/api/health`);
        const data = await response.json();
        if (!active) {
          return;
        }
        setModelReady(Boolean(data?.model?.ready));
      } catch {
        if (!active) {
          return;
        }
        setModelReady(false);
      } finally {
        if (active) {
          setModelStatusLoading(false);
        }
      }
    }

    checkHealth();
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    if (!selectedFile) {
      setPreviewUrl("");
      return;
    }

    const objectUrl = URL.createObjectURL(selectedFile);
    setPreviewUrl(objectUrl);

    return () => {
      URL.revokeObjectURL(objectUrl);
    };
  }, [selectedFile]);

  const subtitle = useMemo(() => {
    if (modelStatusLoading) {
      return "Checking model status...";
    }
    if (modelReady) {
      return "Model ready. Upload an image and run detection.";
    }
    return "Model is missing or empty. Replace the .pt file, then run detection.";
  }, [modelReady, modelStatusLoading]);

  function onPickFile(event) {
    const file = event.target.files?.[0];
    setPrediction(null);
    setResponseTimeMs(null);
    setError("");
    setSelectedFile(file || null);
  }

  async function onAnalyze(event) {
    event.preventDefault();
    if (!selectedFile) {
      setError("Please upload an image first.");
      return;
    }

    const formData = new FormData();
    formData.append("image", selectedFile);

    setError("");
    setLoading(true);
    const start = performance.now();

    try {
      const response = await fetch(`${API_BASE}/api/predict`, {
        method: "POST",
        body: formData
      });
      const data = await response.json();
      if (!response.ok || !data?.success) {
        throw new Error(data?.error || "Prediction failed.");
      }

      setPrediction(data.prediction || null);
      setResponseTimeMs(Math.round(performance.now() - start));
    } catch (err) {
      const message = err instanceof Error ? err.message : "Something went wrong.";
      setError(message);
      setPrediction(null);
      setResponseTimeMs(null);
    } finally {
      setLoading(false);
    }
  }

  const topPrediction = prediction?.detections?.[0];
  const annotatedPreview = prediction?.annotatedImageBase64 || "";
  const confidenceScore = topPrediction?.confidence ?? prediction?.confidence ?? 0;
  const targetPercent = clampPercent(confidenceScore * 100);

  useEffect(() => {
    const from = meterPercentRef.current;
    const to = targetPercent;
    const durationMs = 900;
    let frameId = 0;
    let startTs = 0;

    function step(ts) {
      if (!startTs) {
        startTs = ts;
      }
      const elapsed = ts - startTs;
      const progress = Math.min(1, elapsed / durationMs);
      const eased = 1 - Math.pow(1 - progress, 3);
      const next = from + (to - from) * eased;
      meterPercentRef.current = next;
      setMeterPercent(next);
      if (progress < 1) {
        frameId = requestAnimationFrame(step);
      }
    }

    frameId = requestAnimationFrame(step);
    return () => {
      cancelAnimationFrame(frameId);
    };
  }, [targetPercent]);

  return (
    <div className="page">
      <header className="hero">
        <div>
          <h1>Weed Detection Dashboard</h1>
          <p>{subtitle}</p>
        </div>
        <span className={`status ${modelReady ? "online" : "offline"}`}>
          {modelReady ? "Model Ready" : "Model Not Ready"}
        </span>
      </header>

      <main className="grid">
        <section className="card">
          <h2>Upload Image</h2>
          <form onSubmit={onAnalyze} className="upload-form">
            <label className="upload-field">
              <input type="file" accept="image/*" onChange={onPickFile} />
              <span>{selectedFile ? selectedFile.name : "Choose an image file"}</span>
            </label>
            <button type="submit" disabled={loading}>
              {loading ? "Analyzing..." : "Run Detection"}
            </button>
          </form>

          <div className="confidence-meter">
            <div className="confidence-meter-row">
              <span>Confidence Meter</span>
              <strong>{meterPercent.toFixed(2)}%</strong>
            </div>
            <div className="confidence-meter-track" aria-label="Confidence score bar">
              <div className="confidence-meter-fill" style={{ width: `${meterPercent}%` }} />
            </div>
          </div>

          {previewUrl ? (
            <div className="preview">
              <img src={previewUrl} alt="Uploaded preview" />
            </div>
          ) : (
            <div className="preview placeholder">Image preview will appear here</div>
          )}
        </section>

        <section className="card">
          <h2>Prediction Output</h2>

          {error ? <p className="error">{error}</p> : null}

          {!error && prediction ? (
            <div className="result">
              <div className="result-main">
                <p className="label">Top Prediction</p>
                <h3>{topPrediction?.label || prediction?.label || "Unknown"}</h3>
                <p className="confidence">Confidence: {formatPercent(confidenceScore)}</p>
                <p className="meta">Method: {prediction?.method || "n/a"}</p>
                <p className="meta">Detections: {prediction?.count ?? prediction?.detections?.length ?? 0}</p>
                {responseTimeMs ? <p className="meta">Response time: {responseTimeMs} ms</p> : null}
              </div>

              {annotatedPreview ? (
                <div className="result-image">
                  <img src={annotatedPreview} alt="Tagged weed detection output" />
                </div>
              ) : (
                <p className="meta">Tagged output image is not available for this result.</p>
              )}

              {Array.isArray(prediction?.detections) && prediction.detections.length > 0 ? (
                <ul className="detection-list">
                  {prediction.detections.map((item, index) => (
                    <li key={`${item.label}-${index}`}>
                      <span>{item.label}</span>
                      <span>{formatPercent(item.confidence)}</span>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="meta">No detections returned.</p>
              )}
            </div>
          ) : null}

          {!error && !prediction ? (
            <div className="empty">
              <p>Upload an image and click Run Detection to view results.</p>
            </div>
          ) : null}
        </section>
      </main>
    </div>
  );
}

