import { useState, useMemo, useEffect, useCallback } from "react";

const API_BASE = "http://localhost:8000";

function StatCard({ label, value, accent }) {
  return (
    <div className="rounded-lg border border-slate-700 bg-slate-800 px-4 py-3 text-center">
      <p className={`text-2xl font-bold ${accent || "text-cyan-400"}`}>{value}</p>
      <p className="mt-1 text-xs text-slate-400 uppercase tracking-wide">
        {label}
      </p>
    </div>
  );
}

const DED_BAR_COLORS = {
  0.0: "bg-green-500",
  0.1: "bg-yellow-500",
  0.2: "bg-orange-500",
  0.3: "bg-red-500",
};

function FormScorePanel({ formSummary, timeline }) {
  if (!formSummary || !formSummary.scored_frames) return null;

  const { scored_frames, worst_deduction, avg_deduction, clean_pct, deduction_counts } = formSummary;

  // Grade based on clean percentage
  let grade, gradeColor;
  if (clean_pct >= 90) { grade = "A"; gradeColor = "text-green-400"; }
  else if (clean_pct >= 70) { grade = "B"; gradeColor = "text-cyan-400"; }
  else if (clean_pct >= 50) { grade = "C"; gradeColor = "text-yellow-400"; }
  else if (clean_pct >= 30) { grade = "D"; gradeColor = "text-orange-400"; }
  else { grade = "F"; gradeColor = "text-red-400"; }

  return (
    <div className="rounded-xl border border-slate-700 bg-slate-800/60 p-5 space-y-4">
      <h3 className="text-sm font-semibold text-slate-300">Form Analysis</h3>

      {/* Top-level score */}
      <div className="flex items-center gap-6">
        <div className="text-center">
          <p className={`text-5xl font-black ${gradeColor}`}>{grade}</p>
          <p className="text-xs text-slate-500 mt-1">Grade</p>
        </div>
        <div className="flex-1 grid grid-cols-2 sm:grid-cols-4 gap-3">
          <StatCard label="Clean Frames" value={`${clean_pct}%`} accent="text-green-400" />
          <StatCard label="Avg Deduction" value={avg_deduction.toFixed(2)} />
          <StatCard label="Worst" value={worst_deduction.toFixed(1)} accent="text-red-400" />
          <StatCard label="Scored Frames" value={scored_frames} />
        </div>
      </div>

      {/* Deduction breakdown bar */}
      <div className="space-y-2">
        <p className="text-xs text-slate-500 uppercase tracking-widest">Deduction Breakdown</p>
        <div className="flex h-5 rounded-full overflow-hidden bg-slate-700">
          {["0.0", "0.1", "0.2", "0.3"].map((key) => {
            const count = deduction_counts[key] || 0;
            const pct = (count / scored_frames) * 100;
            if (pct === 0) return null;
            return (
              <div
                key={key}
                className={`${DED_BAR_COLORS[parseFloat(key)]} relative group`}
                style={{ width: `${pct}%` }}
                title={`${key}: ${count} frames (${pct.toFixed(1)}%)`}
              >
                {pct > 8 && (
                  <span className="absolute inset-0 flex items-center justify-center text-[10px] font-bold text-slate-900">
                    {key}
                  </span>
                )}
              </div>
            );
          })}
        </div>
        <div className="flex gap-4 text-[10px] text-slate-500">
          {["0.0", "0.1", "0.2", "0.3"].map((key) => {
            const count = deduction_counts[key] || 0;
            return (
              <span key={key} className="flex items-center gap-1">
                <span className={`inline-block w-2 h-2 rounded-full ${DED_BAR_COLORS[parseFloat(key)]}`} />
                {key}: {count}
              </span>
            );
          })}
        </div>
      </div>

      {/* Deduction timeline sparkline */}
      {timeline && timeline.length > 0 && (
        <div className="space-y-2">
          <p className="text-xs text-slate-500 uppercase tracking-widest">Deduction Timeline</p>
          <div className="flex items-end gap-px h-16 bg-slate-900/50 rounded-lg p-2">
            {timeline.map((pt, i) => {
              const hPct = pt.deduction === 0 ? 5 : (pt.deduction / 0.3) * 100;
              const color = DED_BAR_COLORS[pt.deduction] || "bg-slate-600";
              return (
                <div
                  key={i}
                  className={`flex-1 rounded-t-sm ${color} opacity-80 hover:opacity-100 transition-opacity`}
                  style={{ height: `${hPct}%`, minWidth: "2px" }}
                  title={`Frame ${pt.frame}: ${pt.position} — ded ${pt.deduction} (hip ${pt.hip_avg ?? "?"}° knee ${pt.knee_avg ?? "?"}°)`}
                />
              );
            })}
          </div>
          <div className="flex justify-between text-[10px] text-slate-600">
            <span>Frame {timeline[0].frame}</span>
            <span>Frame {timeline[timeline.length - 1].frame}</span>
          </div>
        </div>
      )}
    </div>
  );
}

const DEDUCTION_BADGE = {
  0.0: "bg-green-500/20 text-green-400 border-green-500/30",
  0.1: "bg-yellow-500/20 text-yellow-400 border-yellow-500/30",
  0.2: "bg-orange-500/20 text-orange-400 border-orange-500/30",
  0.3: "bg-red-500/20 text-red-400 border-red-500/30",
};

const ROLE_LABELS = { entry: "Entry", peak: "Peak", sample: "", exit: "Exit" };

function SegmentCard({ group, onClick }) {
  // Default to peak frame
  const peakIdx = group.findIndex((s) => s.role === "peak");
  const [activeIdx, setActiveIdx] = useState(peakIdx >= 0 ? peakIdx : 0);
  const snap = group[activeIdx] || group[0];
  const badgeClass =
    DEDUCTION_BADGE[snap.deduction] || DEDUCTION_BADGE[0.0];

  return (
    <div
      className="rounded-lg border border-slate-700 bg-slate-800 overflow-hidden cursor-pointer
                 hover:border-cyan-500/50 hover:ring-1 hover:ring-cyan-500/20 transition-all"
    >
      <div className="relative" onClick={() => onClick(activeIdx)}>
        <img
          src={`${API_BASE}${snap.url}`}
          alt={`${snap.position} at frame ${snap.frame}`}
          className="w-full aspect-video object-cover bg-black"
        />
        {/* Role label in corner (only for anchors) */}
        {snap.role && ROLE_LABELS[snap.role] ? (
          <span className="absolute top-2 left-2 text-[10px] font-bold uppercase tracking-wider px-1.5 py-0.5 rounded bg-black/60 text-slate-300">
            {ROLE_LABELS[snap.role]}
          </span>
        ) : null}
      </div>

      {/* Dot indicators */}
      {group.length > 1 && (
        <div className="flex justify-center gap-1.5 pt-2">
          {group.map((s, i) => (
            <button
              key={i}
              onClick={(e) => { e.stopPropagation(); setActiveIdx(i); }}
              className={`w-2 h-2 rounded-full transition-colors ${
                i === activeIdx
                  ? "bg-cyan-400"
                  : "bg-slate-600 hover:bg-slate-400"
              }`}
              title={ROLE_LABELS[s.role] || s.role}
            />
          ))}
        </div>
      )}

      <div className="px-3 py-2.5 space-y-1.5" onClick={() => onClick(activeIdx)}>
        <div className="flex items-center justify-between">
          <span className="text-sm font-semibold text-slate-200">
            {snap.position}
          </span>
          <span
            className={`text-xs font-mono font-bold px-2 py-0.5 rounded border ${badgeClass}`}
          >
            {snap.deduction === 0 ? "0.0" : snap.deduction.toFixed(1)}
          </span>
        </div>
        <p className="text-xs text-slate-500">
          Frame {snap.frame}
          {snap.person_id != null && ` · Person ${snap.person_id}`}
          {snap.description && ` · ${snap.description}`}
        </p>
      </div>
    </div>
  );
}

function Lightbox({ groups, groupIdx, frameIdx, onClose, onPrevGroup, onNextGroup, onSetFrame }) {
  const group = groups[groupIdx] || [];
  const snap = group[frameIdx] || group[0];
  if (!snap) return null;

  const badgeClass =
    DEDUCTION_BADGE[snap.deduction] || DEDUCTION_BADGE[0.0];

  const handleKey = useCallback(
    (e) => {
      if (e.key === "Escape") onClose();
      else if (e.key === "ArrowLeft") onPrevGroup();
      else if (e.key === "ArrowRight") onNextGroup();
    },
    [onClose, onPrevGroup, onNextGroup]
  );

  useEffect(() => {
    document.addEventListener("keydown", handleKey);
    return () => document.removeEventListener("keydown", handleKey);
  }, [handleKey]);

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        className="relative max-w-5xl w-full mx-4 rounded-xl border border-slate-700 bg-slate-900 overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Close button */}
        <button
          onClick={onClose}
          className="absolute top-3 right-3 z-10 rounded-full bg-slate-800/80 p-1.5 text-slate-400 hover:text-white transition-colors"
        >
          <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>

        {/* Role label (only for anchors) */}
        {snap.role && ROLE_LABELS[snap.role] ? (
          <span className="absolute top-3 left-3 z-10 text-xs font-bold uppercase tracking-wider px-2 py-1 rounded bg-black/60 text-slate-300">
            {ROLE_LABELS[snap.role]}
          </span>
        ) : null}

        {/* Image */}
        <img
          src={`${API_BASE}${snap.url}`}
          alt={`${snap.position} at frame ${snap.frame}`}
          className="w-full max-h-[70vh] object-contain bg-black"
        />

        {/* Dot indicators within group */}
        {group.length > 1 && (
          <div className="flex justify-center gap-2 py-2 bg-slate-900">
            {group.map((s, i) => (
              <button
                key={i}
                onClick={() => onSetFrame(i)}
                className={`w-2.5 h-2.5 rounded-full transition-colors ${
                  i === frameIdx
                    ? "bg-cyan-400"
                    : "bg-slate-600 hover:bg-slate-400"
                }`}
                title={ROLE_LABELS[s.role] || s.role}
              />
            ))}
          </div>
        )}

        {/* Info bar */}
        <div className="flex items-center justify-between px-5 py-4">
          <div className="space-y-1">
            <div className="flex items-center gap-3">
              <span className="text-lg font-bold text-slate-100">
                {snap.position}
              </span>
              <span
                className={`text-sm font-mono font-bold px-2.5 py-0.5 rounded border ${badgeClass}`}
              >
                Deduction: {snap.deduction === 0 ? "0.0" : snap.deduction.toFixed(1)}
              </span>
            </div>
            <p className="text-sm text-slate-400">
              Frame {snap.frame}
              {snap.person_id != null && ` · Person ${snap.person_id}`}
              {snap.description && ` · ${snap.description}`}
            </p>
          </div>

          {/* Nav arrows + counter (between groups) */}
          <div className="flex items-center gap-3">
            <button
              onClick={onPrevGroup}
              disabled={groups.length <= 1}
              className="rounded-lg border border-slate-600 p-2 text-slate-400 hover:text-white hover:bg-slate-800 transition-colors disabled:opacity-30"
            >
              <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M15 19l-7-7 7-7" />
              </svg>
            </button>
            <span className="text-xs text-slate-500 font-mono w-12 text-center">
              {groupIdx + 1}/{groups.length}
            </span>
            <button
              onClick={onNextGroup}
              disabled={groups.length <= 1}
              className="rounded-lg border border-slate-600 p-2 text-slate-400 hover:text-white hover:bg-slate-800 transition-colors disabled:opacity-30"
            >
              <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
              </svg>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

function FilterTabs({ label, options, active, onSelect }) {
  return (
    <div className="flex flex-wrap gap-2">
      {options.map(({ key, count }) => {
        const isActive = active === key;
        return (
          <button
            key={key}
            onClick={() => onSelect(key)}
            className={`
              px-3 py-1.5 rounded-lg text-xs font-semibold transition-colors border
              ${isActive
                ? "bg-cyan-500/20 text-cyan-400 border-cyan-500/40"
                : "bg-slate-800 text-slate-400 border-slate-700 hover:border-slate-500 hover:text-slate-300"
              }
            `}
          >
            {key} ({count})
          </button>
        );
      })}
    </div>
  );
}

function SnapshotGallery({ snapshots }) {
  const [posFilter, setPosFilter] = useState("All");
  const [personFilter, setPersonFilter] = useState("All");
  const [lightboxGroupIdx, setLightboxGroupIdx] = useState(null);
  const [lightboxFrameIdx, setLightboxFrameIdx] = useState(0);

  const personIds = useMemo(() => {
    const ids = new Set(snapshots.map((s) => s.person_id).filter((id) => id != null));
    return Array.from(ids).sort((a, b) => a - b);
  }, [snapshots]);

  const positions = useMemo(() => {
    const set = new Set(snapshots.map((s) => s.position));
    return Array.from(set).sort();
  }, [snapshots]);

  const filtered = useMemo(() => {
    let list = snapshots;
    if (posFilter !== "All") list = list.filter((s) => s.position === posFilter);
    if (personFilter !== "All") list = list.filter((s) => s.person_id === personFilter);
    return list;
  }, [snapshots, posFilter, personFilter]);

  // Group filtered snapshots by segment_id
  const groups = useMemo(() => {
    const map = new Map();
    let soloCounter = -1;
    for (const snap of filtered) {
      const sid = snap.segment_id != null ? snap.segment_id : soloCounter--;
      if (!map.has(sid)) map.set(sid, []);
      map.get(sid).push(snap);
    }
    // Sort each group by frame number (backend emits in order, but be safe)
    for (const arr of map.values()) {
      arr.sort((a, b) => a.frame - b.frame);
    }
    return Array.from(map.values());
  }, [filtered]);

  // Count segments for filter display (use all snapshots, not filtered)
  const allGroups = useMemo(() => {
    const map = new Map();
    let soloCounter = -1;
    for (const snap of snapshots) {
      const sid = snap.segment_id != null ? snap.segment_id : soloCounter--;
      if (!map.has(sid)) map.set(sid, []);
      map.get(sid).push(snap);
    }
    return map;
  }, [snapshots]);

  const posOptions = useMemo(() => {
    // Count segments per position
    const posCounts = {};
    for (const [, arr] of allGroups) {
      const pos = arr[0].position;
      posCounts[pos] = (posCounts[pos] || 0) + 1;
    }
    return [
      { key: "All", count: allGroups.size },
      ...positions.map((p) => ({ key: p, count: posCounts[p] || 0 })),
    ];
  }, [allGroups, positions]);

  const personOptions = useMemo(() => {
    const personCounts = {};
    for (const [, arr] of allGroups) {
      const pid = arr[0].person_id;
      if (pid != null) personCounts[pid] = (personCounts[pid] || 0) + 1;
    }
    return [
      { key: "All", count: allGroups.size },
      ...personIds.map((id) => ({ key: id, count: personCounts[id] || 0 })),
    ];
  }, [allGroups, personIds]);

  const personDisplayOptions = useMemo(() =>
    personOptions.map((o) => ({
      ...o,
      key: o.key === "All" ? "All" : o.key,
      label: o.key === "All" ? "All" : `Person ${o.key}`,
    })),
    [personOptions]
  );

  const openLightbox = (groupIdx, frameIdx) => {
    setLightboxGroupIdx(groupIdx);
    setLightboxFrameIdx(frameIdx);
  };
  const closeLightbox = () => {
    setLightboxGroupIdx(null);
    setLightboxFrameIdx(0);
  };
  const peakIdxOf = (group) => {
    const idx = group.findIndex((s) => s.role === "peak");
    return idx >= 0 ? idx : 0;
  };
  const goPrevGroup = () => {
    const newIdx = lightboxGroupIdx > 0 ? lightboxGroupIdx - 1 : groups.length - 1;
    setLightboxGroupIdx(newIdx);
    setLightboxFrameIdx(peakIdxOf(groups[newIdx] || []));
  };
  const goNextGroup = () => {
    const newIdx = lightboxGroupIdx < groups.length - 1 ? lightboxGroupIdx + 1 : 0;
    setLightboxGroupIdx(newIdx);
    setLightboxFrameIdx(peakIdxOf(groups[newIdx] || []));
  };

  const hasMultiplePersons = personIds.length > 1;

  return (
    <div>
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold text-slate-300">
          Position Snapshots ({groups.length} segment{groups.length !== 1 ? "s" : ""})
        </h3>
      </div>

      {/* Filter tabs */}
      <div className="space-y-2 mb-4">
        <FilterTabs
          label="Position"
          options={posOptions}
          active={posFilter}
          onSelect={(v) => { setPosFilter(v); setLightboxGroupIdx(null); }}
        />
        {hasMultiplePersons && (
          <div className="flex flex-wrap gap-2">
            {personDisplayOptions.map(({ key, label, count }) => {
              const isActive = personFilter === key;
              return (
                <button
                  key={key}
                  onClick={() => { setPersonFilter(key); setLightboxGroupIdx(null); }}
                  className={`
                    px-3 py-1.5 rounded-lg text-xs font-semibold transition-colors border
                    ${isActive
                      ? "bg-violet-500/20 text-violet-400 border-violet-500/40"
                      : "bg-slate-800 text-slate-400 border-slate-700 hover:border-slate-500 hover:text-slate-300"
                    }
                  `}
                >
                  {label} ({count})
                </button>
              );
            })}
          </div>
        )}
      </div>

      {/* Segment card grid */}
      {groups.length > 0 ? (
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
          {groups.map((group, i) => (
            <SegmentCard
              key={group[0].segment_id ?? i}
              group={group}
              onClick={(frameIdx) => openLightbox(i, frameIdx)}
            />
          ))}
        </div>
      ) : (
        <p className="text-sm text-slate-500 text-center py-6">
          No snapshots for this filter.
        </p>
      )}

      {/* Lightbox modal */}
      {lightboxGroupIdx !== null && groups[lightboxGroupIdx] && (
        <Lightbox
          groups={groups}
          groupIdx={lightboxGroupIdx}
          frameIdx={lightboxFrameIdx}
          onClose={closeLightbox}
          onPrevGroup={goPrevGroup}
          onNextGroup={goNextGroup}
          onSetFrame={setLightboxFrameIdx}
        />
      )}
    </div>
  );
}

export default function ResultsPanel({ results }) {
  const {
    frame_count,
    fps,
    total_persons_detected,
    processing_time,
    video_url,
    csv_url,
    snapshots = [],
    form_summary,
    deduction_timeline = [],
  } = results;

  return (
    <div className="space-y-5">
      {/* Form analysis (most important — shown first) */}
      <FormScorePanel formSummary={form_summary} timeline={deduction_timeline} />

      {/* Stats grid */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <StatCard label="Frames" value={frame_count} />
        <StatCard label="FPS" value={fps} />
        <StatCard label="Detections" value={total_persons_detected} />
        <StatCard label="Time (s)" value={processing_time} />
      </div>

      {/* Position snapshots with deductions */}
      {snapshots.length > 0 && (
        <SnapshotGallery snapshots={snapshots} />
      )}

      {/* Annotated video player */}
      <div className="rounded-xl border border-slate-700 bg-slate-800/60 overflow-hidden">
        <div className="px-4 pt-3 pb-1">
          <h3 className="text-sm font-semibold text-slate-300">
            Annotated Output
          </h3>
        </div>
        <video src={video_url} controls className="w-full max-h-[450px] bg-black" />
      </div>

      {/* Download buttons */}
      <div className="flex gap-3">
        <a
          href={video_url}
          download
          className="flex-1 rounded-lg bg-cyan-500 py-2.5 text-center text-sm font-semibold text-slate-900 hover:bg-cyan-400 transition-colors"
        >
          Download Video
        </a>
        <a
          href={csv_url}
          download
          className="flex-1 rounded-lg border border-cyan-500 py-2.5 text-center text-sm font-semibold text-cyan-400 hover:bg-cyan-500/10 transition-colors"
        >
          Download CSV
        </a>
      </div>
    </div>
  );
}
