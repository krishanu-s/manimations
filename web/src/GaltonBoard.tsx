import { useRef, useEffect, useState, useCallback } from "react";
import { styles } from "./viz-utils";

// ── Layout ────────────────────────────────────────────────────────────────────

const W = 560;
const H = 560;
const PAD_X = 30;
const PEG_AREA_H = 0.58; // fraction of H for the peg triangle
const CAVITY_H = 0.28; // fraction of H for the cavity bars
const SPAWN_Y = 18;

// ── Physics ───────────────────────────────────────────────────────────────────

const GRAVITY = 0.14;
const RESTITUTION = 0.45; // coefficient of restitution for peg bounces
const MAX_SPEED = 8; // px/frame hard cap to prevent tunneling at large N
const SPAWN_INTERVAL = 8;
const MAX_BALLS = 120;

// ── Colors ────────────────────────────────────────────────────────────────────

const COL = {
  bg: "#f8f9fb",
  peg: "#888",
  pegStroke: "#555",
  ball: "rgba(30, 100, 220, 0.82)",
  ballEdge: "rgba(15, 60, 160, 0.9)",
  barFill: "rgba(40, 110, 230, 0.38)",
  barEdge: "rgba(30, 90, 210, 0.85)",
  theory: "rgba(210, 40, 60, 0.9)",
  wallStroke: "#aaa",
  dim: "#666",
} as const;

type Ball = { x: number; y: number; vx: number; vy: number };

// ── Geometry helpers ──────────────────────────────────────────────────────────

function cavityW(n: number) {
  return (W - 2 * PAD_X) / (n + 1);
}

// Radii scale as 1/√n, capped so pegs and balls always fit within one cavity slot.
function radiiFor(n: number): { pegR: number; ballR: number } {
  const cW = cavityW(n);
  const pegR = Math.max(0.5, Math.min(115, (13 * 12) / n, cW * 0.22));
  const ballR = Math.max(0.8, Math.min(6, 48 / n, cW * 0.26));
  return { pegR, ballR };
}

// Builds the peg triangle for n rows (row k = 0-indexed, k+1 pegs).
// Row k peg j is at x = W/2 + (j − k/2)·cW, placing the bottom row exactly
// on the cavity dividers so each peg routes left→cavity k, right→cavity k+1.
function buildPegs(n: number): { px: Float64Array; py: Float64Array } {
  const total = (n * (n + 1)) / 2;
  const cW = cavityW(n);
  const px = new Float64Array(total);
  const py = new Float64Array(total);
  const yTop = SPAWN_Y + 10;
  const yBot = H * PEG_AREA_H - 8;
  let idx = 0;
  for (let row = 0; row < n; row++) {
    const y =
      n === 1 ? (yTop + yBot) / 2 : yTop + (row / (n - 1)) * (yBot - yTop);

    // for (let j = 0; j <= n; j++) {
    //   px[idx] = (j + ((row + 1) & 1) / 2) * cW;
    //   py[idx] = y;
    //   idx++;
    // }
    for (let j = 0; j <= row; j++) {
      px[idx] = W / 2 + (j - row / 2) * cW;
      py[idx] = y;
      idx++;
    }
  }
  return { px, py };
}

// ── Math helpers ──────────────────────────────────────────────────────────────

const _lf: number[] = [0];
function logFactorial(n: number): number {
  for (let i = _lf.length; i <= n; i++) _lf[i] = _lf[i - 1]! + Math.log(i);
  return _lf[n]!;
}
function binomPMF(n: number, k: number): number {
  if (k < 0 || k > n) return 0;
  return Math.exp(
    logFactorial(n) - logFactorial(k) - logFactorial(n - k) - n * Math.log(2),
  );
}

// ── Physics step ──────────────────────────────────────────────────────────────

function stepBalls(
  balls: Ball[],
  settled: Int32Array,
  n: number,
  pegPx: Float64Array,
  pegPy: Float64Array,
  pegR: number,
  ballR: number,
): Ball[] {
  const cW = cavityW(n);
  const cavTop = H * PEG_AREA_H + 4;
  const floor = H * (PEG_AREA_H + CAVITY_H) - ballR;
  const minD = pegR + ballR;
  const minD2 = minD * minD;
  const surviving: Ball[] = [];

  for (const b of balls) {
    b.vy += GRAVITY;
    b.x += b.vx;
    b.y += b.vy;

    // Peg collisions — inelastic: reflect normal component with restitution e
    for (let i = 0; i < pegPx.length; i++) {
      const dx = b.x - pegPx[i]!;
      const dy = b.y - pegPy[i]!;
      const d2 = dx * dx + dy * dy;
      if (d2 >= minD2 || d2 < 1e-9) continue;
      const d = Math.sqrt(d2);
      const nx = dx / d,
        ny = dy / d;
      const dot = b.vx * nx + b.vy * ny;
      // Reflect with restitution: v' = v − (1+e)·dot·n̂
      b.vx -= (1 + RESTITUTION) * dot * nx;
      b.vy -= (1 + RESTITUTION) * dot * ny;
      // Push out of peg
      b.x += (minD - d) * nx;
      b.y += (minD - d) * ny;
      // Small random nudge so perfectly symmetric paths still diverge
      b.vx += (Math.random() - 0.5) * 0.12;
    }

    // Hard speed cap to prevent tunneling at large N
    const spd = Math.sqrt(b.vx * b.vx + b.vy * b.vy);
    if (spd > MAX_SPEED) {
      b.vx *= MAX_SPEED / spd;
      b.vy *= MAX_SPEED / spd;
    }

    // Cavity walls — inelastic side bounce
    if (b.y > cavTop) {
      const slot = Math.floor((b.x - PAD_X) / cW);
      const wallL = PAD_X + slot * cW;
      const wallR = wallL + cW;
      if (b.x - ballR < wallL) {
        b.x = wallL + ballR;
        b.vx = Math.abs(b.vx) * 0.5;
      }
      if (b.x + ballR > wallR) {
        b.x = wallR - ballR;
        b.vx = -Math.abs(b.vx) * 0.5;
      }
    }

    // Side walls
    if (b.x - ballR < PAD_X) {
      b.x = PAD_X + ballR;
      b.vx = Math.abs(b.vx);
    }
    if (b.x + ballR > W - PAD_X) {
      b.x = W - PAD_X - ballR;
      b.vx = -Math.abs(b.vx);
    }

    // Floor — settle
    if (b.y >= floor) {
      const slot = Math.max(0, Math.min(n, Math.floor((b.x - PAD_X) / cW)));
      if (slot < settled.length) settled[slot]!++;
      continue;
    }

    surviving.push(b);
  }
  return surviving;
}

// ── Renderer ──────────────────────────────────────────────────────────────────

function render(
  ctx: CanvasRenderingContext2D,
  balls: Ball[],
  settled: Int32Array,
  n: number,
  pegPx: Float64Array,
  pegPy: Float64Array,
  pegR: number,
  ballR: number,
  totalDropped: number,
): void {
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = COL.bg;
  ctx.fillRect(0, 0, W, H);

  const cW = cavityW(n);
  const cavTop = H * PEG_AREA_H + 4;
  const floor = H * (PEG_AREA_H + CAVITY_H);

  // Cavity dividers
  ctx.strokeStyle = COL.wallStroke;
  ctx.lineWidth = 1;
  for (let k = 0; k <= n + 1; k++) {
    const x = PAD_X + k * cW;
    ctx.beginPath();
    ctx.moveTo(x, cavTop);
    ctx.lineTo(x, floor);
    ctx.stroke();
  }
  ctx.beginPath();
  ctx.moveTo(PAD_X, floor);
  ctx.lineTo(W - PAD_X, floor);
  ctx.stroke();

  // Histogram bars
  const nSettled = Math.max(
    settled.reduce((a, b) => a + b, 0),
    1,
  );
  let barYMax = 0;
  for (let k = 0; k <= n; k++) {
    const p = binomPMF(n, k);
    if (p > barYMax) barYMax = p;
  }
  barYMax *= 1.35;
  const barH = floor - cavTop;

  for (let k = 0; k <= n; k++) {
    const freq = (settled[k] ?? 0) / nSettled;
    const bx = PAD_X + k * cW;
    const bh = (freq / barYMax) * barH;
    ctx.fillStyle = COL.barFill;
    ctx.fillRect(bx + 0.5, floor - bh, cW - 1, bh);
    ctx.strokeStyle = COL.barEdge;
    ctx.lineWidth = 0.5;
    ctx.strokeRect(bx + 0.5, floor - bh, cW - 1, bh);
  }

  // Theory curve
  ctx.beginPath();
  ctx.strokeStyle = COL.theory;
  ctx.lineWidth = 1.8;
  for (let k = 0; k <= n; k++) {
    const x = PAD_X + (k + 0.5) * cW;
    const y = floor - (binomPMF(n, k) / barYMax) * barH;
    if (k === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // Pegs
  for (let i = 0; i < pegPx.length; i++) {
    ctx.beginPath();
    ctx.arc(pegPx[i]!, pegPy[i]!, pegR, 0, 2 * Math.PI);
    ctx.fillStyle = COL.peg;
    ctx.fill();
    ctx.strokeStyle = COL.pegStroke;
    ctx.lineWidth = 0.6;
    ctx.stroke();
  }

  // In-flight balls
  for (const b of balls) {
    ctx.beginPath();
    ctx.arc(b.x, b.y, ballR, 0, 2 * Math.PI);
    ctx.fillStyle = COL.ball;
    ctx.fill();
    ctx.strokeStyle = COL.ballEdge;
    ctx.lineWidth = 0.8;
    ctx.stroke();
  }

  // X-axis bin labels
  const step = n <= 20 ? 1 : n <= 50 ? 5 : n <= 100 ? 10 : 25;
  ctx.fillStyle = COL.dim;
  ctx.font = "10px system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  for (let k = 0; k <= n; k += step)
    ctx.fillText(String(k), PAD_X + (k + 0.5) * cW, floor + 4);

  // Dropped count
  ctx.fillStyle = COL.dim;
  ctx.font = "11px system-ui, sans-serif";
  ctx.textAlign = "right";
  ctx.textBaseline = "top";
  ctx.fillText(`${totalDropped} dropped`, W - PAD_X, 4);
}

// ── React component ───────────────────────────────────────────────────────────

export default function GaltonBoard() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rafRef = useRef<number>(0);
  const playRef = useRef(true);
  const frameRef = useRef(0);
  const nRef = useRef(10);
  const ballsRef = useRef<Ball[]>([]);
  const settledRef = useRef<Int32Array>(new Int32Array(11));
  const droppedRef = useRef(0);
  const pegsRef = useRef(buildPegs(10));

  const [nState, setNState] = useState(10);
  const [playing, setPlaying] = useState(true);

  function spawnBall(): Ball {
    return {
      x: W / 2 + (Math.random() - 0.5) * 2,
      y: SPAWN_Y,
      vx: (Math.random() - 0.5) * 0.3,
      vy: 0,
    };
  }

  const animate = useCallback(() => {
    const n = nRef.current;
    const { px, py } = pegsRef.current;
    const { pegR, ballR } = radiiFor(n);

    if (playRef.current) {
      frameRef.current++;
      if (
        frameRef.current % SPAWN_INTERVAL === 0 &&
        ballsRef.current.length < MAX_BALLS
      ) {
        ballsRef.current.push(spawnBall());
        droppedRef.current++;
      }
      ballsRef.current = stepBalls(
        ballsRef.current,
        settledRef.current,
        n,
        px,
        py,
        pegR,
        ballR,
      );
    }

    const ctx = canvasRef.current?.getContext("2d");
    if (ctx)
      render(
        ctx,
        ballsRef.current,
        settledRef.current,
        n,
        px,
        py,
        pegR,
        ballR,
        droppedRef.current,
      );

    rafRef.current = requestAnimationFrame(animate);
  }, []);

  useEffect(() => {
    rafRef.current = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(rafRef.current);
  }, [animate]);

  function resetBoard(n: number) {
    nRef.current = n;
    pegsRef.current = buildPegs(n);
    ballsRef.current = [];
    settledRef.current = new Int32Array(n + 1);
    droppedRef.current = 0;
    frameRef.current = 0;
  }

  function handleNChange(e: React.ChangeEvent<HTMLInputElement>) {
    const n = parseInt(e.target.value, 10);
    setNState(n);
    resetBoard(n);
  }

  function reset() {
    resetBoard(nRef.current);
  }

  function togglePlay() {
    playRef.current = !playRef.current;
    setPlaying(playRef.current);
  }

  return (
    <div style={styles.wrapper}>
      <canvas
        ref={canvasRef}
        width={W}
        height={H}
        style={{ ...styles.canvas, display: "block" }}
      />

      <div style={{ ...styles.controls, justifyContent: "center" }}>
        <label style={styles.sliderLabel}>
          <span>
            N = <strong>{nState}</strong>
          </span>
          <input
            type="range"
            min={5}
            max={150}
            step={1}
            value={nState}
            onChange={handleNChange}
            style={styles.slider}
          />
        </label>
        <button onClick={togglePlay} style={styles.btn}>
          {playing ? "Pause" : "Play"}
        </button>
        <button onClick={reset} style={styles.btn}>
          Reset
        </button>
      </div>

      <div style={styles.legend}>
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <svg width="16" height="12">
            <rect
              x="1"
              y="2"
              width="14"
              height="8"
              fill="rgba(40,110,230,0.38)"
              stroke="rgba(30,90,210,0.85)"
              strokeWidth="1"
            />
          </svg>
          <span style={{ fontSize: 13, color: "#444" }}>Simulation</span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <svg width="32" height="12">
            <line
              x1="0"
              y1="6"
              x2="32"
              y2="6"
              stroke="rgba(210,40,60,0.9)"
              strokeWidth="2"
            />
          </svg>
          <span style={{ fontSize: 13, color: "#444" }}>Binomial(N, ½)</span>
        </div>
      </div>
    </div>
  );
}
