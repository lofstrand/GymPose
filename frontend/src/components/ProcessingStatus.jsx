import { useState, useEffect, useRef } from "react";

const API_BASE = "http://localhost:8000";

export default function ProcessingStatus({ onCancel }) {
  const [progress, setProgress] = useState(null);
  const intervalRef = useRef(null);

  useEffect(() => {
    intervalRef.current = setInterval(async () => {
      try {
        const res = await fetch(`${API_BASE}/progress/latest`);
        if (res.ok) {
          setProgress(await res.json());
        }
      } catch {
        // Backend not reachable or no active job yet
      }
    }, 800);

    return () => clearInterval(intervalRef.current);
  }, []);

  const frame = progress?.frame || 0;
  const total = progress?.total_frames || 0;
  const pct = total > 0 ? Math.round((frame / total) * 100) : 0;
  const elapsed = progress?.elapsed || 0;
  const stage = progress?.stage || "starting";

  // Estimate remaining time
  let eta = "";
  if (frame > 0 && total > 0 && elapsed > 0) {
    const remaining = ((elapsed / frame) * (total - frame));
    if (remaining >= 60) {
      eta = `~${Math.round(remaining / 60)}m ${Math.round(remaining % 60)}s remaining`;
    } else {
      eta = `~${Math.round(remaining)}s remaining`;
    }
  }

  const stageLabel =
    stage === "encoding" ? "Re-encoding video for browser playback…" :
    stage === "analyzing" ? "Running pose inference…" :
    "Preparing…";

  return (
    <div className="flex flex-col items-center justify-center rounded-xl border border-slate-700 bg-slate-800/60 py-12 px-8">
      {/* Progress bar */}
      {total > 0 ? (
        <div className="w-full max-w-md mb-6">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-semibold text-slate-200">{stageLabel}</span>
            <span className="text-sm font-mono text-cyan-400">{pct}%</span>
          </div>
          <div className="h-3 w-full rounded-full bg-slate-700 overflow-hidden">
            <div
              className="h-full rounded-full bg-cyan-500 transition-all duration-500 ease-out"
              style={{ width: `${pct}%` }}
            />
          </div>
          <div className="flex items-center justify-between mt-2">
            <span className="text-xs text-slate-500">
              Frame {frame} / {total}
            </span>
            <span className="text-xs text-slate-500">
              {elapsed > 0 && `${elapsed}s elapsed`}
              {eta && ` · ${eta}`}
            </span>
          </div>
        </div>
      ) : (
        <>
          {/* Spinner before first progress update */}
          <div className="relative mb-6">
            <div className="h-14 w-14 rounded-full border-4 border-slate-700" />
            <div className="absolute inset-0 h-14 w-14 rounded-full border-4 border-transparent border-t-cyan-400 animate-spin" />
          </div>
          <p className="text-lg font-semibold text-slate-200">
            Starting analysis&hellip;
          </p>
          <p className="mt-1 text-sm text-slate-500">
            Uploading video and loading model
          </p>
        </>
      )}

      {onCancel && (
        <button
          onClick={onCancel}
          className="mt-6 rounded-lg bg-red-600 px-8 py-2.5 text-sm font-semibold text-white hover:bg-red-500 transition-colors"
        >
          Cancel Analysis
        </button>
      )}
    </div>
  );
}
