/**
 * DistInterpolation.tsx — interpolation between two user-defined distributions A and B.
 *
 * Top row: two interactive panels where the user drags control points to define
 * mean-0 Gaussian-mixture distributions A and B.
 * Bottom panel: PDF of X = √(1−t)·A + √(t)·B (A, B independent), computed via
 * characteristic functions: φ_X(ω) = φ_A(√(1−t)·ω) · φ_B(√(t)·ω).
 */

import { useRef, useEffect, useState, useMemo } from "react";
import {
  PAD,
  COLORS,
  styles,
  makeTransforms,
  drawAxes,
  plotDistribution,
  LegendItem,
} from "./viz-utils";
import {
  evalCharFn,
  fft,
  buildPDF as bPDF,
  N_FFT,
  XMIN,
  XMAX,
  DX,
  XSPACE,
  estimateEntropy,
} from "./pdf-utils";

// ── Constants ──────────────────────────────────────────────────────────────────

const PANEL_W = 340;
const PANEL_H = 260;
const OUT_W = 700;
const OUT_H = 280;

const CTRL_XS: readonly number[] = [-3, -2, -1, 0, 1, 2, 3];
const N_CTRL = 7;
const BASIS_STD = 0.6;
const STANDARD_STDS = Array(N_CTRL).fill(BASIS_STD);
const HIT_R = 14;

const DEFAULT_HEIGHTS_A: readonly number[] = [
  0.05, 0.35, 0.2, 0.1, 0.2, 0.2, 0.05,
];
const DEFAULT_HEIGHTS_B: readonly number[] = [
  0.05, 0.1, 0.15, 0.05, 0.25, 0.3, 0.1,
];

// ── PDF construction ──────────────────────────────────────────────────────────

function buildPDF(heights: readonly number[]): Float64Array {
  let [pdf, norm] = bPDF({
    means: CTRL_XS,
    stds: STANDARD_STDS,
    scales: heights,
  });
  return pdf;
}

// Computes the PDF of X = √(1−t)·A + √(t)·B, A and B independent Gaussian-mixture RVs.
// φ_X(ω) = φ_A(√(1−t)·ω) · φ_B(√(t)·ω)
// p_X(x_k) = FFT[(−1)^j · φ_X(ω_j)][k] / (N·DX)
function computeBlendPDF(
  heightsA: readonly number[],
  heightsB: readonly number[],
  t: number,
): Float64Array {
  const sqA = Math.sqrt(Math.max(0, 1 - t));
  const sqB = Math.sqrt(Math.max(0, t));
  const aRe = new Float64Array(N_FFT);
  const aIm = new Float64Array(N_FFT);
  const gA = { means: CTRL_XS, stds: STANDARD_STDS, scales: heightsA };
  const gB = { means: CTRL_XS, stds: STANDARD_STDS, scales: heightsB };
  for (let j = 0; j < N_FFT; j++) {
    const sj = j <= N_FFT >> 1 ? j : j - N_FFT;
    const omega = (2 * Math.PI * sj) / (N_FFT * DX);
    const [ar, ai] = evalCharFn(gA, sqA * omega);
    const [br, bi] = evalCharFn(gB, sqB * omega);
    const pr = ar * br - ai * bi;
    const pi = ar * bi + ai * br;
    const s = j % 2 === 0 ? 1 : -1;
    aRe[j] = s * pr;
    aIm[j] = s * pi;
  }
  fft(aRe, aIm, false);
  const out = new Float64Array(N_FFT);
  for (let k = 0; k < N_FFT; k++) out[k] = Math.max(0, aRe[k]! / (N_FFT * DX));
  return out;
}

// ── Rendering helpers ──────────────────────────────────────────────────────────

function arrayMax(a: Float64Array): number {
  let m = 0;
  for (let i = 0; i < a.length; i++) if (a[i]! > m) m = a[i]!;
  return m;
}

function niceAxes(peak: number): { ymax: number; yTick: number } {
  const raw = Math.max(peak * 1.3, 0.35);
  const ymax = Math.ceil(raw * 4) / 4;
  return { ymax, yTick: ymax <= 1 ? 0.25 : 0.5 };
}

const COLOR_A = "rgba(30, 90, 210, 0.9)";
const COLOR_B = "rgba(180, 40, 40, 0.9)";
const FILL_A = "rgba(40, 110, 230, 0.18)";
const FILL_B = "rgba(210, 40, 60, 0.18)";
const COLOR_OUT = "rgba(50, 160, 80, 0.9)";
const FILL_OUT = "rgba(50, 200, 100, 0.15)";

function renderInputPanel(
  ctx: CanvasRenderingContext2D,
  pdf: Float64Array,
  heights: readonly number[],
  activeIdx: number | null,
  strokeColor: string,
  fillColor: string,
  W: number,
  H: number,
  outYmax: { value: number },
): void {
  ctx.clearRect(0, 0, W, H);
  const { ymax, yTick } = niceAxes(arrayMax(pdf));
  outYmax.value = ymax;
  const tr = makeTransforms(XMIN, XMAX, ymax, W, H);
  drawAxes(ctx, XMIN, XMAX, ymax, 1, yTick, tr);

  // Distribution fill + stroke
  ctx.fillStyle = fillColor;
  ctx.strokeStyle = strokeColor;
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  ctx.moveTo(tr.cx(XSPACE[0]!), tr.cy(0));
  for (let i = 0; i < N_FFT; i++) ctx.lineTo(tr.cx(XSPACE[i]!), tr.cy(pdf[i]!));
  ctx.lineTo(tr.cx(XSPACE[N_FFT - 1]!), tr.cy(0));
  ctx.closePath();
  ctx.fill();
  ctx.stroke();

  for (let i = 0; i < N_CTRL; i++) {
    const active = i === activeIdx;
    ctx.beginPath();
    ctx.arc(
      tr.cx(CTRL_XS[i]!),
      tr.cy(Math.min(heights[i]!, ymax)),
      active ? 7 : 5,
      0,
      2 * Math.PI,
    );
    ctx.fillStyle = active ? strokeColor : "#333";
    ctx.fill();
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }

  ctx.fillStyle = "rgba(60,60,60,0.8)";
  ctx.font = "12px system-ui, sans-serif";
  ctx.textAlign = "right";
  ctx.textBaseline = "top";
  ctx.fillText(
    `Entropy = ${estimateEntropy(pdf).toFixed(2)}`,
    W - PAD.right - 4,
    PAD.top + 2,
  );
}

function renderOutputPanel(
  ctx: CanvasRenderingContext2D,
  pdfOut: Float64Array,
  pdfA: Float64Array,
  pdfB: Float64Array,
  t: number,
  W: number,
  H: number,
): void {
  ctx.clearRect(0, 0, W, H);
  const peak = Math.max(arrayMax(pdfOut), arrayMax(pdfA), arrayMax(pdfB));
  const { ymax, yTick } = niceAxes(peak);
  const tr = makeTransforms(XMIN, XMAX, ymax, W, H);
  drawAxes(ctx, XMIN, XMAX, ymax, 1, yTick, tr);

  // Draw A and B as thin reference lines
  for (const [pdf, color] of [
    [pdfA, COLOR_A],
    [pdfB, COLOR_B],
  ] as const) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.setLineDash([5, 4]);
    ctx.beginPath();
    for (let i = 0; i < N_FFT; i++) {
      if (i === 0) ctx.moveTo(tr.cx(XSPACE[i]!), tr.cy(pdf[i]!));
      else ctx.lineTo(tr.cx(XSPACE[i]!), tr.cy(pdf[i]!));
    }
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // Draw blended output as filled curve
  ctx.fillStyle = FILL_OUT;
  ctx.strokeStyle = COLOR_OUT;
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  ctx.moveTo(tr.cx(XSPACE[0]!), tr.cy(0));
  for (let i = 0; i < N_FFT; i++)
    ctx.lineTo(tr.cx(XSPACE[i]!), tr.cy(pdfOut[i]!));
  ctx.lineTo(tr.cx(XSPACE[N_FFT - 1]!), tr.cy(0));
  ctx.closePath();
  ctx.fill();
  ctx.stroke();

  ctx.fillStyle = "rgba(60,60,60,0.8)";
  ctx.font = "12px system-ui, sans-serif";
  ctx.textAlign = "right";
  ctx.textBaseline = "top";
  ctx.fillText(`t = ${t.toFixed(2)}`, W - PAD.right - 4, PAD.top + 2);
  ctx.fillText(
    `Entropy = ${estimateEntropy(pdfOut).toFixed(2)}`,
    W - PAD.right - 4,
    PAD.top + 2 + 14,
  );
}

// ── Drag state ─────────────────────────────────────────────────────────────────

interface DragState {
  idx: number;
  startClientY: number;
  startHeights: readonly number[];
  ymax: number;
  rectH: number;
}

function findCtrl(
  clientX: number,
  clientY: number,
  rect: DOMRect,
  heights: readonly number[],
  ymax: number,
  W: number,
  H: number,
): number | null {
  const scaleX = W / rect.width;
  const scaleY = H / rect.height;
  const pw = W - PAD.left - PAD.right;
  const ph = H - PAD.top - PAD.bottom;
  const px = (clientX - rect.left) * scaleX;
  const py = (clientY - rect.top) * scaleY;
  let best: number | null = null;
  let bestD = HIT_R;
  for (let i = 0; i < N_CTRL; i++) {
    const dcx = PAD.left + ((CTRL_XS[i]! - XMIN) / (XMAX - XMIN)) * pw;
    const dcy = H - PAD.bottom - (Math.min(heights[i]!, ymax) / ymax) * ph;
    const d = Math.sqrt((dcx - px) ** 2 + (dcy - py) ** 2);
    if (d < bestD) {
      bestD = d;
      best = i;
    }
  }
  return best;
}

function applyDrag(
  drag: DragState,
  clientY: number,
  H: number,
): readonly number[] | null {
  const ph = H - PAD.top - PAD.bottom;
  const scaleY = H / drag.rectH;
  const delta = ((-(clientY - drag.startClientY) * scaleY) / ph) * drag.ymax;
  const xi = CTRL_XS[drag.idx]!;
  let xsum2 = 0;
  for (let k = 0; k < N_CTRL; k++)
    if (k !== drag.idx) xsum2 += CTRL_XS[k]! ** 2;
  const next = drag.startHeights.map((h0, j) => {
    const vj = j === drag.idx ? 1 : (-xi * CTRL_XS[j]!) / xsum2;
    return h0 + vj * delta;
  });
  if (next.some((v) => v < 0)) return null;
  return next;
}

// ── React component ────────────────────────────────────────────────────────────

export default function DistInterpolation() {
  const refA = useRef<HTMLCanvasElement>(null);
  const refB = useRef<HTMLCanvasElement>(null);
  const refOut = useRef<HTMLCanvasElement>(null);
  const ymaxRefA = useRef<number>(1.0);
  const ymaxRefB = useRef<number>(1.0);
  const dragRefA = useRef<DragState | null>(null);
  const dragRefB = useRef<DragState | null>(null);

  const [heightsA, setHeightsA] =
    useState<readonly number[]>(DEFAULT_HEIGHTS_A);
  const [heightsB, setHeightsB] =
    useState<readonly number[]>(DEFAULT_HEIGHTS_B);
  const [t, setT] = useState(0.5);
  const [activeIdxA, setActiveIdxA] = useState<number | null>(null);
  const [activeIdxB, setActiveIdxB] = useState<number | null>(null);

  const pdfA = useMemo(() => buildPDF(heightsA), [heightsA]);
  const pdfB = useMemo(() => buildPDF(heightsB), [heightsB]);
  const pdfOut = useMemo(
    () => computeBlendPDF(heightsA, heightsB, t),
    [heightsA, heightsB, t],
  );

  useEffect(() => {
    const canvas = refA.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const out = { value: 1.0 };
    renderInputPanel(
      ctx,
      pdfA,
      heightsA,
      activeIdxA,
      COLOR_A,
      FILL_A,
      PANEL_W,
      PANEL_H,
      out,
    );
    ymaxRefA.current = out.value;
  }, [pdfA, heightsA, activeIdxA]);

  useEffect(() => {
    const canvas = refB.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const out = { value: 1.0 };
    renderInputPanel(
      ctx,
      pdfB,
      heightsB,
      activeIdxB,
      COLOR_B,
      FILL_B,
      PANEL_W,
      PANEL_H,
      out,
    );
    ymaxRefB.current = out.value;
  }, [pdfB, heightsB, activeIdxB]);

  useEffect(() => {
    const canvas = refOut.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    renderOutputPanel(ctx, pdfOut, pdfA, pdfB, t, OUT_W, OUT_H);
  }, [pdfOut, pdfA, pdfB, t]);

  // ── Pointer handlers (panel A) ─────────────────────────────────────────────

  function onDownA(e: React.PointerEvent<HTMLCanvasElement>) {
    const rect = e.currentTarget.getBoundingClientRect();
    const idx = findCtrl(
      e.clientX,
      e.clientY,
      rect,
      heightsA,
      ymaxRefA.current,
      PANEL_W,
      PANEL_H,
    );
    if (idx === null) return;
    e.currentTarget.setPointerCapture(e.pointerId);
    dragRefA.current = {
      idx,
      startClientY: e.clientY,
      startHeights: heightsA,
      ymax: ymaxRefA.current,
      rectH: rect.height,
    };
    setActiveIdxA(idx);
  }

  function onMoveA(e: React.PointerEvent<HTMLCanvasElement>) {
    if (!dragRefA.current) return;
    const next = applyDrag(dragRefA.current, e.clientY, PANEL_H);
    if (next) setHeightsA(next);
  }

  function onUpA() {
    dragRefA.current = null;
    setActiveIdxA(null);
  }

  // ── Pointer handlers (panel B) ─────────────────────────────────────────────

  function onDownB(e: React.PointerEvent<HTMLCanvasElement>) {
    const rect = e.currentTarget.getBoundingClientRect();
    const idx = findCtrl(
      e.clientX,
      e.clientY,
      rect,
      heightsB,
      ymaxRefB.current,
      PANEL_W,
      PANEL_H,
    );
    if (idx === null) return;
    e.currentTarget.setPointerCapture(e.pointerId);
    dragRefB.current = {
      idx,
      startClientY: e.clientY,
      startHeights: heightsB,
      ymax: ymaxRefB.current,
      rectH: rect.height,
    };
    setActiveIdxB(idx);
  }

  function onMoveB(e: React.PointerEvent<HTMLCanvasElement>) {
    if (!dragRefB.current) return;
    const next = applyDrag(dragRefB.current, e.clientY, PANEL_H);
    if (next) setHeightsB(next);
  }

  function onUpB() {
    dragRefB.current = null;
    setActiveIdxB(null);
  }

  // ── Render ─────────────────────────────────────────────────────────────────

  return (
    <div style={styles.wrapper}>
      <div
        style={{
          display: "flex",
          gap: 12,
          flexWrap: "wrap",
          justifyContent: "center",
        }}
      >
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            gap: 4,
          }}
        >
          <span style={{ fontSize: 12, color: "#555" }}>
            Distribution A — drag control points
          </span>
          <canvas
            ref={refA}
            width={PANEL_W}
            height={PANEL_H}
            style={{ ...styles.canvas, touchAction: "none" }}
            onPointerDown={onDownA}
            onPointerMove={onMoveA}
            onPointerUp={onUpA}
            onPointerLeave={onUpA}
          />
        </div>
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            gap: 4,
          }}
        >
          <span style={{ fontSize: 12, color: "#555" }}>
            Distribution B — drag control points
          </span>
          <canvas
            ref={refB}
            width={PANEL_W}
            height={PANEL_H}
            style={{ ...styles.canvas, touchAction: "none" }}
            onPointerDown={onDownB}
            onPointerMove={onMoveB}
            onPointerUp={onUpB}
            onPointerLeave={onUpB}
          />
        </div>
      </div>

      <div
        style={{
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          gap: 4,
        }}
      >
        <span style={{ fontSize: 12, color: "#555" }}>
          PDF of X = √(1−t)·A + √(t)·B
        </span>
        <canvas
          ref={refOut}
          width={OUT_W}
          height={OUT_H}
          style={styles.canvas}
        />
      </div>

      <div style={styles.controls}>
        <label style={styles.sliderLabel}>
          <span>
            t = <strong>{t.toFixed(2)}</strong>
          </span>
          <input
            type="range"
            min="0"
            max="1"
            step="0.01"
            value={t}
            onChange={(e) => setT(parseFloat(e.target.value))}
            style={styles.slider}
          />
        </label>
      </div>

      <div style={styles.legend}>
        <LegendItem color={COLOR_A} dash={true} label="Distribution A" />
        <LegendItem color={COLOR_B} dash={true} label="Distribution B" />
        <LegendItem
          color={COLOR_OUT}
          dash={false}
          label="Blend √(1−t)·A + √t·B"
        />
      </div>
    </div>
  );
}
