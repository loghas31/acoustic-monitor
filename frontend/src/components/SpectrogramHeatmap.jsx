// Canvas heatmap of the 40-bin Mel sketch over time. Canvas, not SVG/recharts:
// 288 columns x 40 rows = 11k cells — SVG DOM would crawl, one putImageData won't.

import { useEffect, useRef } from "react";

// Compact perceptual colormap (dark blue -> violet -> orange -> light yellow),
// piecewise-linear approximation of magma.
const STOPS = [
  [0.0, [8, 8, 32]], [0.25, [80, 18, 123]], [0.5, [182, 54, 121]],
  [0.75, [251, 136, 97]], [1.0, [252, 253, 191]],
];

function colorOf(t) {
  for (let i = 1; i < STOPS.length; i++) {
    if (t <= STOPS[i][0]) {
      const [t0, c0] = STOPS[i - 1], [t1, c1] = STOPS[i];
      const u = (t - t0) / (t1 - t0);
      return c0.map((c, k) => Math.round(c + u * (c1[k] - c)));
    }
  }
  return STOPS.at(-1)[1];
}

export default function SpectrogramHeatmap({ columns, height = 180 }) {
  const ref = useRef(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas || !columns?.length) return;
    const nT = columns.length, nM = columns[0].length;
    canvas.width = nT; canvas.height = nM;

    let lo = Infinity, hi = -Infinity;
    for (const col of columns) for (const v of col) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
    const span = hi - lo || 1;

    const ctx = canvas.getContext("2d");
    const img = ctx.createImageData(nT, nM);
    for (let x = 0; x < nT; x++) {
      for (let y = 0; y < nM; y++) {
        const [r, g, b] = colorOf((columns[x][y] - lo) / span);
        // row 0 = lowest Mel bin -> draw at the bottom (audio convention)
        const idx = 4 * ((nM - 1 - y) * nT + x);
        img.data[idx] = r; img.data[idx + 1] = g; img.data[idx + 2] = b; img.data[idx + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
  }, [columns]);

  return (
    <div>
      <canvas ref={ref} style={{ width: "100%", height, imageRendering: "pixelated" }}
              className="rounded-lg ring-1 ring-slate-800" />
      <div className="flex justify-between text-xs text-slate-500 mt-1">
        <span>−24 h</span><span>64 Mel bins (20 Hz – 8 kHz), brighter = more energy</span><span>now</span>
      </div>
    </div>
  );
}
