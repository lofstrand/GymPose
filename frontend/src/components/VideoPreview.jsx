import { useState, useRef, useCallback, useEffect } from "react";

function formatSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export default function VideoPreview({ file, fileUrl, onAnalyze, onRemove, roi, onROIChange }) {
  const containerRef = useRef(null);
  const [drawing, setDrawing] = useState(false);
  const [start, setStart] = useState(null);
  const [current, setCurrent] = useState(null);

  // Convert pixel coords relative to container into 0-1 fractions
  const toFraction = useCallback((px, py) => {
    const el = containerRef.current;
    if (!el) return { x: 0, y: 0 };
    const rect = el.getBoundingClientRect();
    return {
      x: Math.max(0, Math.min(1, (px - rect.left) / rect.width)),
      y: Math.max(0, Math.min(1, (py - rect.top) / rect.height)),
    };
  }, []);

  const handleMouseDown = useCallback((e) => {
    e.preventDefault();
    const pt = toFraction(e.clientX, e.clientY);
    setStart(pt);
    setCurrent(pt);
    setDrawing(true);
  }, [toFraction]);

  const handleMouseMove = useCallback((e) => {
    if (!drawing) return;
    setCurrent(toFraction(e.clientX, e.clientY));
  }, [drawing, toFraction]);

  const handleMouseUp = useCallback(() => {
    if (!drawing || !start || !current) return;
    setDrawing(false);

    const x = Math.min(start.x, current.x);
    const y = Math.min(start.y, current.y);
    const w = Math.abs(current.x - start.x);
    const h = Math.abs(current.y - start.y);

    // Ignore tiny accidental clicks (less than 3% of frame)
    if (w < 0.03 || h < 0.03) {
      return;
    }

    onROIChange({ x, y, w, h });
  }, [drawing, start, current, onROIChange]);

  // Listen globally for mousemove/mouseup so dragging outside the box still works
  useEffect(() => {
    if (!drawing) return;
    const onMove = (e) => handleMouseMove(e);
    const onUp = () => handleMouseUp();
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
  }, [drawing, handleMouseMove, handleMouseUp]);

  // Box style for the drawn rectangle (during drawing or saved ROI)
  const boxStyle = (r) => ({
    position: "absolute",
    left: `${r.x * 100}%`,
    top: `${r.y * 100}%`,
    width: `${r.w * 100}%`,
    height: `${r.h * 100}%`,
    border: "2px solid #22d3ee",
    backgroundColor: "rgba(34, 211, 238, 0.08)",
    pointerEvents: "none",
    borderRadius: "4px",
  });

  // Live draw box
  const liveBox = drawing && start && current
    ? {
        x: Math.min(start.x, current.x),
        y: Math.min(start.y, current.y),
        w: Math.abs(current.x - start.x),
        h: Math.abs(current.y - start.y),
      }
    : null;

  return (
    <div className="rounded-xl border border-slate-700 bg-slate-800/60 overflow-hidden">
      {/* Video + ROI overlay */}
      <div
        ref={containerRef}
        className="relative select-none"
        onMouseDown={handleMouseDown}
      >
        <video
          src={fileUrl}
          controls
          className="w-full max-h-[400px] bg-black pointer-events-auto"
        />

        {/* Transparent overlay to capture mouse on top of video */}
        <div
          className="absolute inset-0 z-10"
          style={{ cursor: drawing ? "crosshair" : "crosshair" }}
          onMouseDown={handleMouseDown}
        />

        {/* Saved ROI */}
        {roi && !drawing && <div style={boxStyle(roi)} />}

        {/* Live drawing box */}
        {liveBox && <div style={boxStyle(liveBox)} />}

        {/* Hint */}
        {!roi && !drawing && (
          <div className="absolute bottom-2 left-1/2 -translate-x-1/2 z-20 px-3 py-1 rounded-full bg-black/70 text-xs text-slate-400">
            Draw a box to set analysis region
          </div>
        )}
      </div>

      {/* File info + actions */}
      <div className="flex items-center justify-between gap-4 px-5 py-4">
        <div className="min-w-0">
          <p className="truncate text-sm font-medium text-slate-200">
            {file.name}
          </p>
          <p className="text-xs text-slate-500">
            {formatSize(file.size)} &middot; {file.type || "video"}
            {roi && (
              <span className="text-cyan-500 ml-2">
                ROI: {(roi.x * 100).toFixed(0)}%,{(roi.y * 100).toFixed(0)}%
                {" "}{(roi.w * 100).toFixed(0)}%x{(roi.h * 100).toFixed(0)}%
              </span>
            )}
          </p>
        </div>

        <div className="flex items-center gap-3 shrink-0">
          {roi && (
            <button
              onClick={() => onROIChange(null)}
              className="rounded-lg border border-slate-600 px-3 py-2 text-sm text-slate-400 hover:text-slate-200 hover:bg-slate-700 transition-colors"
            >
              Clear ROI
            </button>
          )}
          <button
            onClick={onRemove}
            className="rounded-lg border border-slate-600 px-4 py-2 text-sm text-slate-400 hover:text-slate-200 hover:bg-slate-700 transition-colors"
          >
            Remove
          </button>
          <button
            onClick={onAnalyze}
            className="rounded-lg bg-cyan-500 px-5 py-2 text-sm font-semibold text-slate-900 hover:bg-cyan-400 transition-colors"
          >
            Analyze Poses
          </button>
        </div>
      </div>
    </div>
  );
}
