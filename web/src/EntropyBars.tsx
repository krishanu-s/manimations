/**
 * EntropyBars.tsx — interactive entropy visualization.
 *
 * The x-axis [0, 1] is partitioned into n intervals of widths p₁, …, pₙ.
 * Above each interval sits a rectangle of height −log(pᵢ).  The total
 * shaded area equals the Shannon entropy H = −Σ pᵢ log pᵢ (in nats).
 *
 * Controls:
 *   • Drag the vertical dividers to redistribute probability mass.
 *   • +/− buttons add or remove an interval (split widest / drop last).
 */

import { useRef, useEffect, useState } from "react";
import { PAD, COLORS, styles } from "./viz-utils";

// ─── Constants ────────────────────────────────────────────────────────────────

const CANVAS_W = 660;
const CANVAS_H = 400;
const YMAX = 5; // fixed y-axis ceiling; bars reaching it are clipped
const MIN_WIDTH = 0.02; // minimum interval width (so max bar ≈ -ln(0.02) ≈ 3.91 < YMAX)
const HIT_PX = 10; // pixel radius for divider hit-testing

// ─── Domain helpers ───────────────────────────────────────────────────────────

function uniformDividers(n: number): number[] {
  return Array.from({ length: n - 1 }, (_, i) => (i + 1) / n);
}

function getProbs(dividers: readonly number[]): number[] {
  const edges = [0, ...dividers, 1];
  return Array.from(
    { length: edges.length - 1 },
    (_, i) => edges[i + 1]! - edges[i]!,
  );
}

function shannonEntropy(probs: readonly number[]): number {
  return probs.reduce(
    (s, p) => (p > 0 ? s - (p * Math.log(p)) / Math.log(2) : s),
    0,
  );
}

// ─── Rendering ───────────────────────────────────────────────────────────────

function render(
  ctx: CanvasRenderingContext2D,
  dividers: readonly number[],
  hovered: number | null,
  W: number,
  H: number,
): void {
  const probs = getProbs(dividers);
  const edges = [0, ...dividers, 1];
  const pw = W - PAD.left - PAD.right;
  const ph = H - PAD.top - PAD.bottom;

  const cx = (x: number) => PAD.left + x * pw;
  const cy = (y: number) =>
    H - PAD.bottom - (Math.min(Math.max(y, 0), YMAX) / YMAX) * ph;

  ctx.clearRect(0, 0, W, H);

  // ── Bars ──────────────────────────────────────────────────────────────────
  for (let i = 0; i < probs.length; i++) {
    const p = probs[i]!;
    const barTop = p > 0 ? Math.min(-Math.log(p), YMAX) : YMAX;
    const bx = cx(edges[i]!);
    const bw = cx(edges[i + 1]!) - bx;
    const by = cy(barTop);
    const bh = cy(0) - by;

    ctx.fillStyle = COLORS.distFill;
    ctx.fillRect(bx, by, bw, bh);
    ctx.strokeStyle = COLORS.distStroke;
    ctx.lineWidth = 1.5;
    ctx.strokeRect(bx, by, bw, bh);
  }

  // ── Grid ──────────────────────────────────────────────────────────────────
  ctx.lineWidth = 1;
  ctx.strokeStyle = COLORS.grid;
  for (let yi = 1; yi < YMAX; yi++) {
    ctx.beginPath();
    ctx.moveTo(cx(0), cy(yi));
    ctx.lineTo(cx(1), cy(yi));
    ctx.stroke();
  }
  for (const xv of [0.2, 0.4, 0.6, 0.8]) {
    ctx.beginPath();
    ctx.moveTo(cx(xv), cy(0));
    ctx.lineTo(cx(xv), cy(YMAX));
    ctx.stroke();
  }

  // ── Axes ──────────────────────────────────────────────────────────────────
  ctx.strokeStyle = COLORS.axis;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(cx(0), cy(0));
  ctx.lineTo(cx(1), cy(0)); // x-axis
  ctx.moveTo(cx(0), cy(0));
  ctx.lineTo(cx(0), cy(YMAX)); // y-axis
  ctx.stroke();

  // ── x-axis ticks and labels ───────────────────────────────────────────────
  ctx.fillStyle = COLORS.tick;
  ctx.font = "13px system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  ctx.lineWidth = 1;
  for (let xi = 0; xi <= 5; xi++) {
    const xv = xi * 0.2;
    ctx.strokeStyle = COLORS.axis;
    ctx.beginPath();
    ctx.moveTo(cx(xv), cy(0));
    ctx.lineTo(cx(xv), cy(0) + 5);
    ctx.stroke();
    ctx.fillText(
      xv === 0 ? "0" : xv === 1 ? "1" : xv.toFixed(1),
      cx(xv),
      cy(0) + 7,
    );
  }

  // ── y-axis ticks and labels ───────────────────────────────────────────────
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (let yi = 1; yi <= YMAX; yi++) {
    ctx.strokeStyle = COLORS.axis;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(cx(0), cy(yi));
    ctx.lineTo(cx(0) - 5, cy(yi));
    ctx.stroke();
    ctx.fillText(String(yi), cx(0) - 8, cy(yi));
  }

  // ── Axis name labels ──────────────────────────────────────────────────────
  ctx.fillStyle = COLORS.label;
  ctx.font = "16px system-ui, sans-serif";
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  ctx.fillText("p", cx(1) + 6, cy(0));
  ctx.save();
  ctx.translate(cx(0) - 14, PAD.top - 14 + ph / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = "center";
  ctx.textBaseline = "bottom";
  ctx.fillText("-log p", 0, 0);
  ctx.restore();

  // ── Divider lines and drag handles ────────────────────────────────────────
  for (let i = 0; i < dividers.length; i++) {
    const xv = dividers[i]!;
    const isH = i === hovered;

    ctx.strokeStyle = isH ? "#222" : "#666";
    ctx.lineWidth = isH ? 2.5 : 1.5;
    ctx.setLineDash(isH ? [] : [5, 3]);
    ctx.beginPath();
    ctx.moveTo(cx(xv), cy(YMAX));
    ctx.lineTo(cx(xv), cy(0));
    ctx.stroke();
    ctx.setLineDash([]);

    // Small handle knob below the x-axis
    const kx = cx(xv);
    const ky = cy(0) + 6;
    ctx.fillStyle = isH ? "#222" : "#777";
    ctx.beginPath();
    ctx.roundRect(kx - 5, ky, 10, 14, 3);
    ctx.fill();
  }

  // ── Entropy annotation ────────────────────────────────────────────────────
  const entropy = shannonEntropy(probs);
  ctx.fillStyle = "rgba(40, 40, 40, 0.8)";
  ctx.font = "13px system-ui, sans-serif";
  ctx.textAlign = "right";
  ctx.textBaseline = "top";
  ctx.fillText(
    `entropy = ${entropy.toFixed(4)}`,
    W - PAD.right - 4,
    PAD.top + 2,
  );
}

// ─── React component ─────────────────────────────────────────────────────────

interface DragState {
  index: number;
  startClientX: number;
  startDivX: number; // divider's domain position at drag start
  snapDividers: number[]; // full dividers array frozen at drag start
}

export default function EntropyBars() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [dividers, setDividers] = useState<number[]>(uniformDividers(3));
  const [hovered, setHovered] = useState<number | null>(null);
  const dragRef = useRef<DragState | null>(null);

  // Re-render whenever dividers or hover state changes
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    render(ctx, dividers, hovered, CANVAS_W, CANVAS_H);
  }, [dividers, hovered]);

  // Convert clientX → domain position in [0, 1]
  function clientToDomain(clientX: number, rect: DOMRect): number {
    const scaleX = CANVAS_W / rect.width;
    const px = (clientX - rect.left) * scaleX;
    const pw = CANVAS_W - PAD.left - PAD.right;
    return (px - PAD.left) / pw;
  }

  // Return the index of whichever divider is within HIT_PX of clientX, or null
  function findDivider(
    clientX: number,
    rect: DOMRect,
    divs: readonly number[],
  ): number | null {
    const scaleX = CANVAS_W / rect.width;
    const px = (clientX - rect.left) * scaleX;
    const pw = CANVAS_W - PAD.left - PAD.right;
    const cxFn = (x: number) => PAD.left + x * pw;
    let best: number | null = null;
    let bestDist = HIT_PX;
    for (let i = 0; i < divs.length; i++) {
      const dist = Math.abs(cxFn(divs[i]!) - px);
      if (dist < bestDist) {
        bestDist = dist;
        best = i;
      }
    }
    return best;
  }

  function handlePointerDown(e: React.PointerEvent<HTMLCanvasElement>) {
    const rect = e.currentTarget.getBoundingClientRect();
    const idx = findDivider(e.clientX, rect, dividers);
    if (idx === null) return;
    e.currentTarget.setPointerCapture(e.pointerId);
    dragRef.current = {
      index: idx,
      startClientX: e.clientX,
      startDivX: dividers[idx]!,
      snapDividers: [...dividers],
    };
  }

  function handlePointerMove(e: React.PointerEvent<HTMLCanvasElement>) {
    const rect = e.currentTarget.getBoundingClientRect();

    if (!dragRef.current) {
      const idx = findDivider(e.clientX, rect, dividers);
      setHovered(idx);
      e.currentTarget.style.cursor = idx !== null ? "ew-resize" : "default";
      return;
    }

    const drag = dragRef.current;
    const delta =
      clientToDomain(e.clientX, rect) - clientToDomain(drag.startClientX, rect);
    const newX = drag.startDivX + delta;

    const divs = [...drag.snapDividers];
    const lo = drag.index > 0 ? divs[drag.index - 1]! + MIN_WIDTH : MIN_WIDTH;
    const hi =
      drag.index < divs.length - 1
        ? divs[drag.index + 1]! - MIN_WIDTH
        : 1 - MIN_WIDTH;
    divs[drag.index] = Math.max(lo, Math.min(hi, newX));
    setDividers(divs);
  }

  function handlePointerUp() {
    dragRef.current = null;
  }

  function handlePointerLeave(e: React.PointerEvent<HTMLCanvasElement>) {
    if (!dragRef.current) {
      setHovered(null);
      e.currentTarget.style.cursor = "default";
    }
  }

  function increaseN() {
    // Split the widest interval at its midpoint
    const probs = getProbs(dividers);
    const maxIdx = probs.indexOf(Math.max(...probs));
    const edges = [0, ...dividers, 1];
    const mid = (edges[maxIdx]! + edges[maxIdx + 1]!) / 2;
    const next = [...dividers];
    next.splice(maxIdx, 0, mid);
    setDividers(next);
  }

  function decreaseN() {
    if (dividers.length === 0) return;
    setDividers(dividers.slice(0, -1));
  }

  const n = dividers.length + 1;

  return (
    <div style={styles.wrapper}>
      <canvas
        ref={canvasRef}
        width={CANVAS_W}
        height={CANVAS_H}
        style={{ ...styles.canvas, touchAction: "none" }}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onPointerLeave={handlePointerLeave}
      />
      <div style={styles.controls}>
        <span style={{ fontSize: "13px", minWidth: "5em" }}>n = {n}</span>
        <button onClick={decreaseN} disabled={n <= 1} style={styles.btn}>
          − Remove
        </button>
        <button onClick={increaseN} style={styles.btn}>
          + Add
        </button>
      </div>
    </div>
  );
}
