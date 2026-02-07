export default function SettingsPanel({ settings, onChange }) {
  const update = (key, value) => onChange({ ...settings, [key]: value });

  return (
    <aside className="w-64 shrink-0 border-r border-slate-700/60 bg-slate-800/40 px-5 py-8 flex flex-col gap-7">
      <h2 className="text-xs font-semibold uppercase tracking-widest text-slate-500">
        Settings
      </h2>

      {/* Model selector */}
      <label className="block">
        <span className="text-sm text-slate-300">Model</span>
        <select
          value={settings.model}
          onChange={(e) => update("model", e.target.value)}
          className="mt-2 w-full rounded-lg border border-slate-600 bg-slate-900 px-3 py-2 text-sm text-slate-200 focus:border-cyan-500 focus:outline-none"
        >
          <optgroup label="YOLO26 (latest)">
            <option value="yolo26n-pose.pt">YOLO26n — Nano (fastest)</option>
            <option value="yolo26s-pose.pt">YOLO26s — Small</option>
            <option value="yolo26m-pose.pt">YOLO26m — Medium</option>
            <option value="yolo26l-pose.pt">YOLO26l — Large</option>
            <option value="yolo26x-pose.pt">YOLO26x — XLarge (most accurate)</option>
          </optgroup>
          <optgroup label="YOLO11">
            <option value="yolo11n-pose.pt">YOLO11n — Nano</option>
            <option value="yolo11s-pose.pt">YOLO11s — Small (balanced)</option>
            <option value="yolo11m-pose.pt">YOLO11m — Medium</option>
            <option value="yolo11l-pose.pt">YOLO11l — Large</option>
            <option value="yolo11x-pose.pt">YOLO11x — XLarge</option>
          </optgroup>
          <optgroup label="YOLOv8">
            <option value="yolov8n-pose.pt">YOLOv8n — Nano</option>
            <option value="yolov8s-pose.pt">YOLOv8s — Small</option>
            <option value="yolov8m-pose.pt">YOLOv8m — Medium</option>
            <option value="yolov8l-pose.pt">YOLOv8l — Large</option>
            <option value="yolov8x-pose.pt">YOLOv8x — XLarge</option>
          </optgroup>
        </select>
      </label>

      {/* Target position */}
      <label className="block">
        <span className="text-sm text-slate-300">Target Position</span>
        <select
          value={settings.targetPosition}
          onChange={(e) => update("targetPosition", e.target.value)}
          className="mt-2 w-full rounded-lg border border-slate-600 bg-slate-900 px-3 py-2 text-sm text-slate-200 focus:border-cyan-500 focus:outline-none"
        >
          <option value="auto">Auto-detect</option>
          <option value="pike">Pike</option>
          <option value="tuck">Tuck</option>
          <option value="layout">Layout (Straight)</option>
        </select>
        <p className="mt-1 text-xs text-slate-500">
          {settings.targetPosition === "auto"
            ? "Deductions scored per detected position"
            : `All frames scored against ${settings.targetPosition} criteria`}
        </p>
      </label>

      {/* Confidence threshold */}
      <label className="block">
        <span className="text-sm text-slate-300">Confidence Threshold</span>
        <div className="mt-2 flex items-center gap-3">
          <input
            type="range"
            min="0.1"
            max="1"
            step="0.05"
            value={settings.confThreshold}
            onChange={(e) => update("confThreshold", parseFloat(e.target.value))}
            className="flex-1 accent-cyan-500"
          />
          <span className="w-10 text-right text-sm font-mono text-cyan-400">
            {settings.confThreshold.toFixed(2)}
          </span>
        </div>
      </label>

      {/* Max persons */}
      <label className="block">
        <span className="text-sm text-slate-300">Max Persons</span>
        <input
          type="number"
          min="1"
          max="10"
          value={settings.maxPersons}
          onChange={(e) =>
            update("maxPersons", Math.min(10, Math.max(1, parseInt(e.target.value) || 1)))
          }
          className="mt-2 w-full rounded-lg border border-slate-600 bg-slate-900 px-3 py-2 text-sm text-slate-200 focus:border-cyan-500 focus:outline-none"
        />
      </label>

      {/* Overlay toggles */}
      <div className="space-y-3">
        <span className="text-xs font-semibold uppercase tracking-widest text-slate-500">
          Overlays
        </span>
        {[
          { key: "showSkeleton", label: "Skeleton Lines" },
          { key: "showAngles", label: "Angles" },
          { key: "showPosition", label: "Position" },
          { key: "showDeduction", label: "Deduction" },
        ].map(({ key, label }) => (
          <label key={key} className="flex items-center justify-between cursor-pointer">
            <span className="text-sm text-slate-300">{label}</span>
            <button
              type="button"
              role="switch"
              aria-checked={settings[key]}
              onClick={() => update(key, !settings[key])}
              className={`
                relative inline-flex h-6 w-11 items-center rounded-full transition-colors
                ${settings[key] ? "bg-cyan-500" : "bg-slate-600"}
              `}
            >
              <span
                className={`
                  inline-block h-4 w-4 rounded-full bg-white transition-transform
                  ${settings[key] ? "translate-x-6" : "translate-x-1"}
                `}
              />
            </button>
          </label>
        ))}
      </div>

      {/* Detect inverted toggle */}
      <label className="flex items-center justify-between cursor-pointer">
        <span className="text-sm text-slate-300">Detect Inverted</span>
        <button
          type="button"
          role="switch"
          aria-checked={settings.detectInverted}
          onClick={() => update("detectInverted", !settings.detectInverted)}
          className={`
            relative inline-flex h-6 w-11 items-center rounded-full transition-colors
            ${settings.detectInverted ? "bg-cyan-500" : "bg-slate-600"}
          `}
        >
          <span
            className={`
              inline-block h-4 w-4 rounded-full bg-white transition-transform
              ${settings.detectInverted ? "translate-x-6" : "translate-x-1"}
            `}
          />
        </button>
      </label>

      {/* Divider + info */}
      <div className="mt-auto border-t border-slate-700 pt-4">
        <p className="text-xs text-slate-600 leading-relaxed">
          Model: {settings.model.replace('.pt', '')}<br />
          17 COCO keypoints<br />
          8 joint angles tracked
        </p>
      </div>
    </aside>
  );
}
