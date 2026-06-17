/**
 * CLTInterpolation.tsx — interpolation between a user-defined distribution and a Gaussian
 * via the Central Limit Theorem.
 *
 * Left panel: user drags control points to define a symmetric (mean-0) distribution X.
 * Right panel: PDF of Y = √(1−t)·(X₁+…+Xₙ)/√n + √t·G, where G ~ N(0, σ²),
 * computed via the characteristic function: φ_Y(ω) = φ_X(√(1−t)·ω/√n)^n · exp(−½tσ²ω²).
 */

import { useRef, useEffect, useState, useMemo } from "react";
import {
  PAD, COLORS, styles,
  makeTransforms, drawAxes, plotDistribution, plotDashedCurve, LegendItem,
} from "./viz-utils";

// ── Grid ───────────────────────────────────────────────────────────────────────

const N_FFT = 512;          // must be a power of 2
const XMIN = -6.0;
const XMAX = 6.0;
const DX = (XMAX - XMIN) / N_FFT;
const XSPACE = Object.freeze(
  Array.from({ length: N_FFT }, (_, k) => XMIN + k * DX),
) as readonly number[];

const PANEL_W = 380;
const PANEL_H = 300;

// Symmetric control points: dragging index i also moves index (6−i)
const CTRL_XS: readonly number[] = [-3, -2, -1, 0, 1, 2, 3];
const N_CTRL = 7;
const BASIS_STD = 0.6;   // std of each Gaussian basis function
const HIT_R = 14;        // pixel hit radius for control points

const DEFAULT_HEIGHTS: readonly number[] = [0.05, 0.4, 1.2, 0.05, 1.2, 0.4, 0.05];

// ── FFT (Cooley-Tukey radix-2, in-place) ─────────────────────────────────────

function fft(re: Float64Array, im: Float64Array, invert: boolean): void {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      let t = re[i]!; re[i] = re[j]!; re[j] = t;
      t = im[i]!; im[i] = im[j]!; im[j] = t;
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (invert ? 1 : -1) * (2 * Math.PI / len);
    const wr = Math.cos(ang), wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let j = 0; j < (len >> 1); j++) {
        const u = i + j, v = i + j + (len >> 1);
        const vr = re[v]! * cr - im[v]! * ci;
        const vi = re[v]! * ci + im[v]! * cr;
        re[v] = re[u]! - vr; im[v] = im[u]! - vi;
        re[u] = re[u]! + vr; im[u] = im[u]! + vi;
        const nc = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = nc;
      }
    }
  }
  if (invert) for (let i = 0; i < n; i++) { re[i]! /= n; im[i]! /= n; }
}

// ── PDF construction ───────────────────────────────────────────────────────────

function gaussPDF(x: number, mu: number, sigma: number): number {
  const z = (x - mu) / sigma;
  return Math.exp(-0.5 * z * z) / (sigma * Math.sqrt(2 * Math.PI));
}

function buildPDF(heights: readonly number[]): Float64Array {
  const pdf = new Float64Array(N_FFT);
  let norm = 0;
  for (let k = 0; k < N_FFT; k++) {
    const x = XSPACE[k]!;
    let v = 0;
    for (let i = 0; i < N_CTRL; i++) v += heights[i]! * gaussPDF(x, CTRL_XS[i]!, BASIS_STD);
    pdf[k] = v;
    norm += v;
  }
  norm *= DX;
  if (norm > 1e-15) for (let k = 0; k < N_FFT; k++) pdf[k]! /= norm;
  return pdf;
}

function computeVariance(pdf: Float64Array): number {
  let v = 0;
  for (let k = 0; k < N_FFT; k++) v += XSPACE[k]! ** 2 * pdf[k]!;
  return v * DX;
}

// ── Characteristic function φ_X via FFT ───────────────────────────────────────
// φ_X(ω_j) = DX · (−1)^j · conj(FFT[p_X][j])
// Proof: φ(ω_j) = DX · e^{iω_j·x₀} · conj(FFT[p][j])
//        with x₀ = XMIN = −L and ω_j = 2πj/(N·DX) = πj/L, so ω_j·x₀ = −πj.

function computeCharFn(pdf: Float64Array): [Float64Array, Float64Array] {
  const re = Float64Array.from(pdf);
  const im = new Float64Array(N_FFT);
  fft(re, im, false);
  const cRe = new Float64Array(N_FFT);
  const cIm = new Float64Array(N_FFT);
  for (let j = 0; j < N_FFT; j++) {
    const s = j % 2 === 0 ? 1 : -1;
    cRe[j] = DX * s * re[j]!;
    cIm[j] = DX * s * (-im[j]!);  // conjugate: negate imaginary part
  }
  return [cRe, cIm];
}

// Linear interpolation of φ_X at a fractional signed frequency index
function interpChar(cRe: Float64Array, cIm: Float64Array, si: number): [number, number] {
  const idx = ((si % N_FFT) + N_FFT) % N_FFT;
  const i0 = Math.floor(idx);
  const i1 = (i0 + 1) % N_FFT;
  const f = idx - i0;
  return [
    cRe[i0]! + f * (cRe[i1]! - cRe[i0]!),
    cIm[i0]! + f * (cIm[i1]! - cIm[i0]!),
  ];
}

// Complex power via polar form: (a + ib)^n
function cplxPow(a: number, b: number, n: number): [number, number] {
  const r = Math.sqrt(a * a + b * b);
  if (r < 1e-30) return [0, 0];
  const th = Math.atan2(b, a) * n;
  return [r ** n * Math.cos(th), r ** n * Math.sin(th)];
}

// ── Output PDF via inverse FFT ─────────────────────────────────────────────────
// φ_Y(ω) = φ_X(α·ω)^n · exp(−½·t·σ²·ω²), where α = √(1−t)/√n.
// p_Y(x_k) = FFT[(−1)^j · φ_Y(ω_j)][k] / (N·DX).

function computeOutputPDF(
  cRe: Float64Array,
  cIm: Float64Array,
  sigma2: number,
  n: number,
  t: number,
): Float64Array {
  const alpha = Math.sqrt(Math.max(0, 1 - t)) / Math.sqrt(n);
  const aRe = new Float64Array(N_FFT);
  const aIm = new Float64Array(N_FFT);
  for (let j = 0; j < N_FFT; j++) {
    const sj = j <= (N_FFT >> 1) ? j : j - N_FFT;   // signed frequency index
    const [xr, xi] = interpChar(cRe, cIm, sj * alpha);
    const [pr, pi] = cplxPow(xr, xi, n);
    const omega = 2 * Math.PI * sj / (N_FFT * DX);
    const gf = Math.exp(-0.5 * t * sigma2 * omega * omega);
    const s = j % 2 === 0 ? 1 : -1;
    aRe[j] = s * pr * gf;
    aIm[j] = s * pi * gf;
  }
  fft(aRe, aIm, false);  // forward FFT gives p_Y up to the 1/(N·DX) factor
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
  const ymax = Math.ceil(raw * 4) / 4;  // round up to nearest 0.25 (exact in float)
  return { ymax, yTick: ymax <= 1 ? 0.25 : 0.5 };
}

function renderLeft(
  ctx: CanvasRenderingContext2D,
  pdf: Float64Array,
  heights: readonly number[],
  activeIdx: number | null,
  W: number,
  H: number,
  outYmax: { value: number },
): void {
  ctx.clearRect(0, 0, W, H);
  const { ymax, yTick } = niceAxes(arrayMax(pdf));
  outYmax.value = ymax;
  const tr = makeTransforms(XMIN, XMAX, ymax, W, H);
  drawAxes(ctx, XMIN, XMAX, ymax, 1, yTick, tr);
  plotDistribution(ctx, XSPACE, pdf, tr);

  // Control point dots
  for (let i = 0; i < N_CTRL; i++) {
    const active = i === activeIdx;
    ctx.beginPath();
    ctx.arc(tr.cx(CTRL_XS[i]!), tr.cy(Math.min(heights[i]!, ymax)), active ? 7 : 5, 0, 2 * Math.PI);
    ctx.fillStyle = active ? COLORS.targetStroke : "#333";
    ctx.fill();
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }
}

function renderRight(
  ctx: CanvasRenderingContext2D,
  pdf: Float64Array,
  sigma2: number,
  n: number,
  t: number,
  W: number,
  H: number,
): void {
  ctx.clearRect(0, 0, W, H);
  const std = Math.sqrt(Math.max(sigma2, 1e-9));
  const peakGauss = gaussPDF(0, 0, std);
  const { ymax, yTick } = niceAxes(Math.max(arrayMax(pdf), peakGauss));
  const tr = makeTransforms(XMIN, XMAX, ymax, W, H);
  drawAxes(ctx, XMIN, XMAX, ymax, 1, yTick, tr);
  plotDashedCurve(ctx, XSPACE, x => gaussPDF(x, 0, std), tr);
  plotDistribution(ctx, XSPACE, pdf, tr);
  ctx.fillStyle = "rgba(60,60,60,0.8)";
  ctx.font = "12px system-ui, sans-serif";
  ctx.textAlign = "right";
  ctx.textBaseline = "top";
  ctx.fillText(`t = ${t.toFixed(2)},  n = ${n}`, W - PAD.right - 4, PAD.top + 2);
}

// ── React component ────────────────────────────────────────────────────────────

interface DragState {
  idx: number;
  startClientY: number;
  startHeight: number;
  ymax: number;
  rectH: number;
}

export default function CLTInterpolation() {
  const leftRef = useRef<HTMLCanvasElement>(null);
  const rightRef = useRef<HTMLCanvasElement>(null);
  const ymaxRef = useRef<number>(1.0);
  const [heights, setHeights] = useState<readonly number[]>(DEFAULT_HEIGHTS);
  const [t, setT] = useState(0);
  const [n, setN] = useState(1);
  const [activeIdx, setActiveIdx] = useState<number | null>(null);
  const dragRef = useRef<DragState | null>(null);

  const pdf = useMemo(() => buildPDF(heights), [heights]);
  const sigma2 = useMemo(() => computeVariance(pdf), [pdf]);
  const charFn = useMemo(() => computeCharFn(pdf), [pdf]);
  const outputPdf = useMemo(
    () => computeOutputPDF(charFn[0], charFn[1], sigma2, n, t),
    [charFn, sigma2, n, t],
  );

  useEffect(() => {
    const canvas = leftRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const out = { value: 1.0 };
    renderLeft(ctx, pdf, heights, activeIdx, PANEL_W, PANEL_H, out);
    ymaxRef.current = out.value;
  }, [pdf, heights, activeIdx]);

  useEffect(() => {
    const canvas = rightRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    renderRight(ctx, outputPdf, sigma2, n, t, PANEL_W, PANEL_H);
  }, [outputPdf, sigma2, n, t]);

  // ── Pointer events for the left (interactive) panel ───────────────────────

  function findCtrl(clientX: number, clientY: number, rect: DOMRect): number | null {
    const scaleX = PANEL_W / rect.width;
    const scaleY = PANEL_H / rect.height;
    const pw = PANEL_W - PAD.left - PAD.right;
    const ph = PANEL_H - PAD.top - PAD.bottom;
    const ymax = ymaxRef.current;
    const px = (clientX - rect.left) * scaleX;
    const py = (clientY - rect.top) * scaleY;
    let best: number | null = null;
    let bestD = HIT_R;
    for (let i = 0; i < N_CTRL; i++) {
      const dcx = PAD.left + ((CTRL_XS[i]! - XMIN) / (XMAX - XMIN)) * pw;
      const dcy = PANEL_H - PAD.bottom - (Math.min(heights[i]!, ymax) / ymax) * ph;
      const d = Math.sqrt((dcx - px) ** 2 + (dcy - py) ** 2);
      if (d < bestD) { bestD = d; best = i; }
    }
    return best;
  }

  function handlePointerDown(e: React.PointerEvent<HTMLCanvasElement>) {
    const rect = e.currentTarget.getBoundingClientRect();
    const idx = findCtrl(e.clientX, e.clientY, rect);
    if (idx === null) return;
    e.currentTarget.setPointerCapture(e.pointerId);
    dragRef.current = {
      idx,
      startClientY: e.clientY,
      startHeight: heights[idx]!,
      ymax: ymaxRef.current,
      rectH: rect.height,
    };
    setActiveIdx(idx);
  }

  function handlePointerMove(e: React.PointerEvent<HTMLCanvasElement>) {
    if (!dragRef.current) return;
    const drag = dragRef.current;
    const ph = PANEL_H - PAD.top - PAD.bottom;
    const scaleY = PANEL_H / drag.rectH;
    const delta = -(e.clientY - drag.startClientY) * scaleY / ph * drag.ymax;
    const newH = Math.max(0, drag.startHeight + delta);
    setHeights(prev => {
      const next = [...prev];
      next[drag.idx] = newH;
      const mirror = 6 - drag.idx;
      if (mirror !== drag.idx) next[mirror] = newH;
      return next;
    });
  }

  function handlePointerUp() {
    dragRef.current = null;
    setActiveIdx(null);
  }

  return (
    <div style={styles.wrapper}>
      <div style={{ display: "flex", gap: 12, flexWrap: "wrap", justifyContent: "center" }}>
        <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 4 }}>
          <span style={{ fontSize: 12, color: "#555" }}>Distribution X — drag control points</span>
          <canvas
            ref={leftRef}
            width={PANEL_W}
            height={PANEL_H}
            style={{ ...styles.canvas, touchAction: "none" }}
            onPointerDown={handlePointerDown}
            onPointerMove={handlePointerMove}
            onPointerUp={handlePointerUp}
            onPointerLeave={handlePointerUp}
          />
        </div>
        <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 4 }}>
          <span style={{ fontSize: 12, color: "#555" }}>PDF of Y = √(1−t)·Sₙ + √t·G</span>
          <canvas
            ref={rightRef}
            width={PANEL_W}
            height={PANEL_H}
            style={styles.canvas}
          />
        </div>
      </div>

      <div style={styles.controls}>
        <label style={styles.sliderLabel}>
          <span>t = <strong>{t.toFixed(2)}</strong></span>
          <input
            type="range" min="0" max="1" step="0.01" value={t}
            onChange={e => setT(parseFloat(e.target.value))}
            style={styles.slider}
          />
        </label>
        <span style={{ fontSize: 13, minWidth: "4em" }}>n = {n}</span>
        <button onClick={() => setN(v => Math.max(1, v - 1))} disabled={n <= 1} style={styles.btn}>n −</button>
        <button onClick={() => setN(v => Math.min(100, v + 1))} style={styles.btn}>n +</button>
      </div>

      <div style={styles.legend}>
        <LegendItem color={COLORS.distStroke} dash={false} label="Current distribution" />
        <LegendItem
          color={COLORS.targetStroke}
          dash={true}
          label={`Target N(0, σ² = ${sigma2.toFixed(3)})`}
        />
      </div>
    </div>
  );
}
