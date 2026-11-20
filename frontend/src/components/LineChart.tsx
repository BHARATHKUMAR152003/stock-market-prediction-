import { useId, useMemo, useState, type MouseEvent } from 'react';

export interface Series {
  name: string;
  color: string;
  values: (number | null)[];
  dashed?: boolean;
}

interface Props {
  title: string;
  labels: string[];
  series: Series[];
  marker?: { index: number; label: string };
  height?: number;
  format?: (v: number) => string;
}

const W = 760;
const PAD = { top: 12, right: 12, bottom: 26, left: 56 };

/** Dependency-free SVG line chart with a hover readout; null values leave gaps. */
export default function LineChart({ title, labels, series, marker, height = 260, format = (v) => v.toFixed(2) }: Props) {
  const [hover, setHover] = useState<number | null>(null);
  const titleId = useId();
  const H = height;
  const n = labels.length;

  const { lo, hi } = useMemo(() => {
    const all = series.flatMap((s) => s.values.filter((v): v is number => v !== null && Number.isFinite(v)));
    if (!all.length) return { lo: 0, hi: 1 };
    const min = Math.min(...all);
    const max = Math.max(...all);
    const pad = (max - min) * 0.05 || 1;
    return { lo: min - pad, hi: max + pad };
  }, [series]);

  const x = (i: number) => PAD.left + (n <= 1 ? 0 : (i / (n - 1)) * (W - PAD.left - PAD.right));
  const y = (v: number) => PAD.top + (1 - (v - lo) / (hi - lo)) * (H - PAD.top - PAD.bottom);

  const path = (values: (number | null)[]) => {
    let d = '';
    let pen = false;
    values.forEach((v, i) => {
      if (v === null || !Number.isFinite(v)) {
        pen = false;
        return;
      }
      d += `${pen ? 'L' : 'M'}${x(i).toFixed(1)},${y(v).toFixed(1)}`;
      pen = true;
    });
    return d;
  };

  const ticks = [0, 0.25, 0.5, 0.75, 1].map((t) => lo + t * (hi - lo));
  const xTicks = n > 1 ? [0, Math.floor((n - 1) / 2), n - 1] : [0];

  const onMove = (e: MouseEvent<SVGSVGElement>) => {
    const box = e.currentTarget.getBoundingClientRect();
    const px = ((e.clientX - box.left) / box.width) * W;
    const i = Math.round(((px - PAD.left) / (W - PAD.left - PAD.right)) * (n - 1));
    setHover(i >= 0 && i < n ? i : null);
  };

  return (
    <figure className="m-0">
      <figcaption id={titleId} className="mb-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-fog-300">
        <span className="font-semibold text-fog-100">{title}</span>
        {series.map((s) => (
          <span key={s.name} className="inline-flex items-center gap-1.5">
            <span aria-hidden="true" className="inline-block h-0.5 w-4" style={{ background: s.color }} />
            {s.name}
          </span>
        ))}
      </figcaption>
      <div className="relative">
        <svg
          viewBox={`0 0 ${W} ${H}`}
          className="h-auto w-full"
          role="img"
          aria-labelledby={titleId}
          onMouseMove={onMove}
          onMouseLeave={() => setHover(null)}
        >
          {ticks.map((t) => (
            <g key={t}>
              <line x1={PAD.left} x2={W - PAD.right} y1={y(t)} y2={y(t)} stroke="#252d38" />
              <text x={PAD.left - 6} y={y(t) + 4} textAnchor="end" fontSize="11" fill="#8b97a6">
                {format(t)}
              </text>
            </g>
          ))}
          {xTicks.map((i) => (
            <text key={i} x={x(i)} y={H - 6} textAnchor={i === 0 ? 'start' : i === n - 1 ? 'end' : 'middle'} fontSize="11" fill="#8b97a6">
              {labels[i]}
            </text>
          ))}
          {marker && marker.index >= 0 && marker.index < n && (
            <g>
              <line x1={x(marker.index)} x2={x(marker.index)} y1={PAD.top} y2={H - PAD.bottom} stroke="#8b97a6" strokeDasharray="4 4" />
              <text x={x(marker.index) + 4} y={PAD.top + 10} fontSize="11" fill="#aab4c0">
                {marker.label}
              </text>
            </g>
          )}
          {series.map((s) => (
            <path key={s.name} d={path(s.values)} fill="none" stroke={s.color} strokeWidth="1.6" strokeDasharray={s.dashed ? '5 4' : undefined} />
          ))}
          {hover !== null && <line x1={x(hover)} x2={x(hover)} y1={PAD.top} y2={H - PAD.bottom} stroke="#cfd6de" strokeOpacity="0.4" />}
        </svg>
        {hover !== null && (
          <div className="pointer-events-none absolute right-2 top-2 rounded-lg border border-ink-600 bg-ink-850/95 px-3 py-2 text-xs">
            <div className="font-medium text-fog-100">{labels[hover]}</div>
            {series.map((s) =>
              s.values[hover] === null || s.values[hover] === undefined ? null : (
                <div key={s.name} style={{ color: s.color }}>
                  {s.name}: {format(s.values[hover] as number)}
                </div>
              ),
            )}
          </div>
        )}
      </div>
    </figure>
  );
}
