import { useState, useRef, useCallback } from "react";

const ACCEPTED = ".mp4,.mov,.avi,.webm";

export default function UploadZone({ onFileSelect, disabled }) {
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef(null);

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    setDragging(true);
  }, []);

  const handleDragLeave = useCallback((e) => {
    e.preventDefault();
    setDragging(false);
  }, []);

  const handleDrop = useCallback(
    (e) => {
      e.preventDefault();
      setDragging(false);
      const f = e.dataTransfer.files[0];
      if (f) onFileSelect(f);
    },
    [onFileSelect]
  );

  const handleChange = useCallback(
    (e) => {
      const f = e.target.files[0];
      if (f) onFileSelect(f);
    },
    [onFileSelect]
  );

  return (
    <div
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      className={`
        relative flex flex-col items-center justify-center rounded-xl border-2 border-dashed
        px-8 py-16 text-center transition-all cursor-pointer
        ${
          dragging
            ? "border-cyan-400 bg-cyan-400/10"
            : "border-slate-600 bg-slate-800/50 hover:border-slate-500 hover:bg-slate-800"
        }
        ${disabled ? "pointer-events-none opacity-50" : ""}
      `}
      onClick={() => inputRef.current?.click()}
    >
      {/* Upload icon */}
      <svg
        className={`mb-4 h-12 w-12 ${dragging ? "text-cyan-400" : "text-slate-500"}`}
        fill="none"
        viewBox="0 0 24 24"
        stroke="currentColor"
        strokeWidth={1.5}
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5"
        />
      </svg>

      <p className="text-sm text-slate-300">
        <span className="font-semibold text-cyan-400">Drop a video here</span>{" "}
        or click to browse
      </p>
      <p className="mt-1 text-xs text-slate-500">MP4, MOV, AVI, WebM</p>

      <input
        ref={inputRef}
        type="file"
        accept={ACCEPTED}
        onChange={handleChange}
        className="hidden"
      />
    </div>
  );
}
