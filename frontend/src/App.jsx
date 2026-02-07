import { useState, useCallback, useRef } from "react";
import UploadZone from "./components/UploadZone";
import VideoPreview from "./components/VideoPreview";
import ProcessingStatus from "./components/ProcessingStatus";
import ResultsPanel from "./components/ResultsPanel";
import SettingsPanel from "./components/SettingsPanel";

const API_BASE = "http://localhost:8000";

export default function App() {
  const [file, setFile] = useState(null);
  const [fileUrl, setFileUrl] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [settings, setSettings] = useState({
    confThreshold: 0.10,
    maxPersons: 5,
    showSkeleton: false,
    showAngles: false,
    showPosition: true,
    showDeduction: true,
    detectInverted: true,
    model: "yolo26n-pose.pt",
    targetPosition: "auto",
  });

  const [roi, setROI] = useState(null); // {x, y, w, h} as 0-1 fractions or null
  const abortRef = useRef(null);
  const jobIdRef = useRef(null);

  const handleFileSelect = useCallback((f) => {
    setFile(f);
    setFileUrl(URL.createObjectURL(f));
    setResults(null);
    setError(null);
    setROI(null);
  }, []);

  const handleAnalyze = useCallback(async () => {
    if (!file) return;
    setAnalyzing(true);
    setError(null);
    setResults(null);
    jobIdRef.current = null;

    const controller = new AbortController();
    abortRef.current = controller;

    const formData = new FormData();
    formData.append("video", file);
    formData.append("conf_threshold", settings.confThreshold);
    formData.append("max_persons", settings.maxPersons);
    formData.append("show_skeleton", settings.showSkeleton ? "true" : "false");
    formData.append("show_angles", settings.showAngles ? "true" : "false");
    formData.append("show_position", settings.showPosition ? "true" : "false");
    formData.append("show_deduction", settings.showDeduction ? "true" : "false");
    formData.append("detect_inverted", settings.detectInverted ? "true" : "false");
    formData.append("model_name", settings.model);
    formData.append("target_position", settings.targetPosition);
    if (roi) {
      formData.append("roi_x", roi.x);
      formData.append("roi_y", roi.y);
      formData.append("roi_w", roi.w);
      formData.append("roi_h", roi.h);
    }

    try {
      const res = await fetch(`${API_BASE}/analyze`, {
        method: "POST",
        body: formData,
        signal: controller.signal,
      });

      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail || `Server error ${res.status}`);
      }

      const data = await res.json();
      jobIdRef.current = data.job_id;
      // Prefix result URLs with API base
      data.video_url = `${API_BASE}${data.video_url}`;
      data.csv_url = `${API_BASE}${data.csv_url}`;
      setResults(data);
    } catch (err) {
      if (err.name === "AbortError") {
        // User cancelled — not an error
        return;
      }
      if (err.name === "TypeError" && err.message === "Failed to fetch") {
        setError(
          "Cannot reach the backend server. Make sure it is running on port 8000."
        );
      } else {
        setError(err.message);
      }
    } finally {
      abortRef.current = null;
      setAnalyzing(false);
    }
  }, [file, settings, roi]);

  const handleCancel = useCallback(async () => {
    // Abort the fetch request
    if (abortRef.current) {
      abortRef.current.abort();
      abortRef.current = null;
    }
    // Tell backend to stop processing
    try {
      await fetch(`${API_BASE}/cancel/latest`, { method: "POST" });
    } catch {
      // Best-effort
    }
    setAnalyzing(false);
  }, []);

  const handleReset = useCallback(() => {
    setFile(null);
    if (fileUrl) URL.revokeObjectURL(fileUrl);
    setFileUrl(null);
    setResults(null);
    setError(null);
    setROI(null);
  }, [fileUrl]);

  return (
    <div className="flex min-h-screen">
      {/* Sidebar */}
      <SettingsPanel settings={settings} onChange={setSettings} />

      {/* Main content */}
      <main className="flex-1 flex flex-col items-center px-6 py-8 overflow-y-auto">
        {/* Header */}
        <div className="mb-8 text-center">
          <h1 className="text-4xl font-bold tracking-tight">
            <span className="text-cyan-400">Gym</span>Pose
          </h1>
          <p className="mt-2 text-slate-400 text-sm">
            AI-powered gymnastics pose detection &amp; analysis
          </p>
        </div>

        <div className="w-full max-w-3xl space-y-6">
          {/* Upload zone (always visible when no results) */}
          {!results && !analyzing && (
            <>
              {!file ? (
                <UploadZone
                  onFileSelect={handleFileSelect}
                  disabled={analyzing}
                />
              ) : (
                <VideoPreview
                  file={file}
                  fileUrl={fileUrl}
                  onAnalyze={handleAnalyze}
                  onRemove={handleReset}
                  analyzing={analyzing}
                  roi={roi}
                  onROIChange={setROI}
                />
              )}
            </>
          )}

          {/* Processing indicator */}
          {analyzing && <ProcessingStatus onCancel={handleCancel} />}

          {/* Error */}
          {error && (
            <div className="rounded-lg border border-red-500/30 bg-red-500/10 px-5 py-4 text-red-300 text-sm">
              <span className="font-semibold">Error:</span> {error}
            </div>
          )}

          {/* Results */}
          {results && (
            <>
              <ResultsPanel results={results} />
              <button
                onClick={handleReset}
                className="mt-4 w-full rounded-lg border border-slate-600 py-2.5 text-sm text-slate-300 hover:bg-slate-800 transition-colors"
              >
                Analyze Another Video
              </button>
            </>
          )}
        </div>
      </main>
    </div>
  );
}
