import { useRef, useEffect, useState, useCallback } from "react";
import { styles, LegendItem, COLORS } from "./viz-utils";

// ── Layout constants ──────────────────────────────────────────────────────────

const BOX_W = 400;
const BOX_H = 400;

const DEFAULT_T = 1;
const T_MIN = 0.25;
const T_MAX = 4;
const T_STEP = 0.05;

const HC_W = 280;
const HC_H = 400;
const HIST_H = 125;
const H_YOFFS = [5, 140, 275] as const;
const HP = { t: 20, r: 14, b: 32, l: 30 } as const;
const N_BINS = 60;

const DEFAULT_N = 400;
const N_MIN = 60;
const N_MAX = 1000;
const N_STEP = 10;

// ── Colors ────────────────────────────────────────────────────────────────────

const COL = {
  particle: "rgba(30, 100, 220, 0.78)",
  particleEdge: "rgba(15, 60, 160, 0.90)",
  barFill: "rgba(40, 110, 230, 0.35)",
  barEdge: "rgba(30, 90, 210, 0.82)",
  theory: "rgba(210, 40, 60, 0.90)",
  axis: "#555",
  text: "#333",
  dim: "#666",
  bg: "#f8f9fb",
} as const;

type Particle = { x: number; y: number; vx: number; vy: number };

function radiusFor(n: number) {
  return 40 / Math.sqrt(n);
}

// ── Initialization ────────────────────────────────────────────────────────────

// Produces a vector of the given length whose angle is chosen uniformly in [0, 2π]
function randomRadialVector(r: number): [number, number] {
  const θ = Math.random() * 2 * Math.PI;
  return [r * Math.cos(θ), r * Math.sin(θ)];
}

// Produces a vector of the given length whose angle is chosen uniformly in [π/4, 3π/4, 5π/4, 7π/4]
function randomDiagonalVector(r: number): [number, number] {
  const θ = ((Math.floor(4 * Math.random()) + 0.5) * Math.PI) / 2;
  return [r * Math.cos(θ), r * Math.sin(θ)];
}

function initParticles(n: number, r: number, speed0: number): Particle[] {
  const cols = Math.ceil(Math.sqrt(n));
  const rows = Math.ceil(n / cols);
  const cw = BOX_W / cols;
  const ch = BOX_H / rows;
  const ps: Particle[] = [];
  let k = 0;
  outer: for (let ri = 0; ri < rows; ri++) {
    for (let ci = 0; ci < cols; ci++) {
      if (k++ >= n) break outer;
      const x = Math.min(BOX_W - r, Math.max(r, (ci + 0.5) * cw));
      const y = Math.min(BOX_H - r, Math.max(r, (ri + 0.5) * ch));
      // Initialization option 1: each particle has velocity with magnitude speed0 in a random direction.
      const θ = Math.random() * 2 * Math.PI;
      const [vx, vy] = randomDiagonalVector(speed0);
      ps.push({ x, y, vx: vx, vy: vy });
    }
  }
  return ps;
}

// ── Physics ───────────────────────────────────────────────────────────────────

function physicsStep(ps: Particle[], r: number): void {
  const n = ps.length;
  const dmin = 2 * r;
  const dmin2 = dmin * dmin;

  for (let i = 0; i < n; i++) {
    ps[i]!.x += ps[i]!.vx;
    ps[i]!.y += ps[i]!.vy;
  }

  for (let i = 0; i < n; i++) {
    const p = ps[i]!;
    if (p.x < r) {
      p.x = 2 * r - p.x;
      p.vx = Math.abs(p.vx);
    }
    if (p.x > BOX_W - r) {
      p.x = 2 * (BOX_W - r) - p.x;
      p.vx = -Math.abs(p.vx);
    }
    if (p.y < r) {
      p.y = 2 * r - p.y;
      p.vy = Math.abs(p.vy);
    }
    if (p.y > BOX_H - r) {
      p.y = 2 * (BOX_H - r) - p.y;
      p.vy = -Math.abs(p.vy);
    }
  }

  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const a = ps[i]!,
        b = ps[j]!;
      const dx = b.x - a.x,
        dy = b.y - a.y;
      const d2 = dx * dx + dy * dy;
      if (d2 >= dmin2 || d2 < 1e-12) continue;
      const d = Math.sqrt(d2);
      const nx = dx / d,
        ny = dy / d;
      const dvn = (a.vx - b.vx) * nx + (a.vy - b.vy) * ny;
      if (dvn <= 0) continue;
      a.vx -= dvn * nx;
      a.vy -= dvn * ny;
      b.vx += dvn * nx;
      b.vy += dvn * ny;
      const half = 0.5 * (dmin - d);
      a.x -= half * nx;
      a.y -= half * ny;
      b.x += half * nx;
      b.y += half * ny;
    }
  }
}

// ── Renderers ─────────────────────────────────────────────────────────────────

function drawSim(
  ctx: CanvasRenderingContext2D,
  ps: Particle[],
  r: number,
): void {
  ctx.fillStyle = COL.bg;
  ctx.fillRect(0, 0, BOX_W, BOX_H);
  for (const p of ps) {
    ctx.beginPath();
    ctx.arc(p.x, p.y, r, 0, 2 * Math.PI);
    ctx.fillStyle = COL.particle;
    ctx.fill();
    ctx.strokeStyle = COL.particleEdge;
    ctx.lineWidth = 0.8;
    ctx.stroke();
  }
}

function drawOneHist(
  ctx: CanvasRenderingContext2D,
  yOff: number,
  values: readonly number[],
  xMin: number,
  xMax: number,
  title: string,
  xlabel: string,
  theoryFn: (x: number) => number,
): void {
  const { t, r, b, l } = HP;
  const pw = HC_W - l - r;
  const ph = HIST_H - t - b;
  const range = xMax - xMin;
  const bw = range / N_BINS;

  ctx.fillStyle = "#fff";
  ctx.fillRect(0, yOff, HC_W, HIST_H);

  const counts = new Array<number>(N_BINS).fill(0);
  for (const v of values) {
    const idx = Math.floor(((v - xMin) / range) * N_BINS);
    if (idx >= 0 && idx < N_BINS) counts[idx]!++;
  }
  const n = Math.max(values.length, 1);
  const dens = counts.map((c) => c / (n * bw));

  let yMax = 1e-9;
  for (let i = 0; i <= 120; i++) {
    const tv = theoryFn(xMin + (i / 120) * range);
    if (tv > yMax) yMax = tv;
  }
  yMax *= 1.4;

  const cx = (x: number) => l + ((x - xMin) / range) * pw;
  const cy = (v: number) => yOff + t + (1 - Math.min(v / yMax, 1)) * ph;
  const y0 = yOff + t + ph;

  const bpx = pw / N_BINS;
  for (let i = 0; i < N_BINS; i++) {
    const bx = cx(xMin + i * bw);
    const bh = (dens[i]! / yMax) * ph;
    ctx.fillStyle = COL.barFill;
    ctx.fillRect(bx, y0 - bh, bpx - 0.5, bh);
    ctx.strokeStyle = COL.barEdge;
    ctx.lineWidth = 0.5;
    ctx.strokeRect(bx, y0 - bh, bpx - 0.5, bh);
  }

  ctx.beginPath();
  ctx.strokeStyle = COL.theory;
  ctx.lineWidth = 1.8;
  for (let i = 0; i <= 200; i++) {
    const x = xMin + (i / 200) * range;
    if (i === 0) ctx.moveTo(cx(x), cy(theoryFn(x)));
    else ctx.lineTo(cx(x), cy(theoryFn(x)));
  }
  ctx.stroke();

  ctx.strokeStyle = COL.axis;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(l, yOff + t);
  ctx.lineTo(l, y0);
  ctx.lineTo(l + pw, y0);
  ctx.stroke();

  ctx.fillStyle = COL.text;
  ctx.font = "bold 11px system-ui, sans-serif";
  ctx.textAlign = "left";
  ctx.textBaseline = "top";
  ctx.fillText(title, l, yOff + 4);

  ctx.font = "10px system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  for (let i = 0; i <= 5; i++) {
    const v = xMin + (i / 5) * range;
    const x = cx(v);
    ctx.strokeStyle = COL.axis;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x, y0);
    ctx.lineTo(x, y0 + 3);
    ctx.stroke();
    ctx.fillStyle = COL.dim;
    ctx.fillText(v.toFixed(1), x, y0 + 5);
  }

  ctx.fillStyle = COL.dim;
  ctx.font = "10px system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  ctx.fillText(xlabel, l + pw / 2, y0 + 17);

  ctx.font = "9px system-ui, sans-serif";
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (let i = 1; i <= 2; i++) {
    const v = (i / 2) * yMax;
    const y = cy(v);
    ctx.strokeStyle = COL.axis;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(l, y);
    ctx.lineTo(l - 3, y);
    ctx.stroke();
    ctx.fillStyle = COL.dim;
    ctx.fillText(v.toFixed(2), l - 5, y);
  }

  ctx.fillStyle = COL.dim;
  ctx.font = "10px system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "bottom";
  ctx.fillText("p", l, yOff + t - 2);
}

function drawHistograms(ctx: CanvasRenderingContext2D, ps: Particle[]): void {
  ctx.fillStyle = COL.bg;
  ctx.fillRect(0, 0, HC_W, HC_H);

  const vxArr = ps.map((p) => p.vx);
  const vyArr = ps.map((p) => p.vy);
  const keArr = ps.map((p) => 0.5 * (p.vx * p.vx + p.vy * p.vy));

  const kT = keArr.reduce((s, k) => s + k, 0) / ps.length;
  const sigma = Math.sqrt(Math.max(kT, 1e-9));

  // Fix the x-scale of the vx and vy plots
  // const vRange = Math.max(4 * sigma, 0.1);
  const vRange = 4.0;

  // Fix the x-scale of the kinetic energy plot
  // const keMax = Math.max(5 * kT, 0.1);
  const keMax = 5.0;

  const gaussFn = (v: number) =>
    (1 / (Math.sqrt(2 * Math.PI) * sigma)) * Math.exp((-v * v) / (2 * kT));
  const expFn = (ke: number) => (ke >= 0 ? (1 / kT) * Math.exp(-ke / kT) : 0);

  drawOneHist(ctx, H_YOFFS[0], keArr, 0, keMax, "kinetic energy", "KE", expFn);
  drawOneHist(
    ctx,
    H_YOFFS[1],
    vxArr,
    -vRange,
    vRange,
    "vx velocities",
    "vx",
    gaussFn,
  );
  drawOneHist(
    ctx,
    H_YOFFS[2],
    vyArr,
    -vRange,
    vRange,
    "vy velocities",
    "vy",
    gaussFn,
  );
}

// ── React component ───────────────────────────────────────────────────────────

export default function GasSimulation() {
  const simRef = useRef<HTMLCanvasElement>(null);
  const histRef = useRef<HTMLCanvasElement>(null);
  const rafRef = useRef<number>(0);
  const playRef = useRef(true);

  // N, R, T and speed0 are held in refs so the animation loop always sees
  // current values without needing to restart the rAF loop on slider changes.
  const nRef = useRef(DEFAULT_N);
  const rRef = useRef(radiusFor(DEFAULT_N));
  const tRef = useRef(DEFAULT_T);
  const speed0Ref = useRef(Math.sqrt(DEFAULT_T));
  const psRef = useRef<Particle[]>(
    initParticles(nRef.current, rRef.current, speed0Ref.current),
  );

  const [nState, setNState] = useState(DEFAULT_N);
  const [tState, setTState] = useState(DEFAULT_T);
  const [playing, setPlaying] = useState(true);

  const animate = useCallback(() => {
    if (playRef.current) physicsStep(psRef.current, rRef.current);
    const sCtx = simRef.current?.getContext("2d");
    if (sCtx) drawSim(sCtx, psRef.current, rRef.current);
    const hCtx = histRef.current?.getContext("2d");
    if (hCtx) drawHistograms(hCtx, psRef.current);
    rafRef.current = requestAnimationFrame(animate);
  }, []);

  useEffect(() => {
    rafRef.current = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(rafRef.current);
  }, [animate]);

  function handleNChange(e: React.ChangeEvent<HTMLInputElement>) {
    const n = parseInt(e.target.value, 10);
    const r = radiusFor(n);
    nRef.current = n;
    rRef.current = r;
    psRef.current = initParticles(n, r, speed0Ref.current);
    setNState(n);
  }

  function handleTChange(e: React.ChangeEvent<HTMLInputElement>) {
    const tNew = parseFloat(e.target.value);
    const tOld = tRef.current;
    const scale = Math.sqrt(tNew / tOld);
    for (const p of psRef.current) {
      p.vx *= scale;
      p.vy *= scale;
    }
    speed0Ref.current = Math.sqrt(tNew);
    tRef.current = tNew;
    setTState(tNew);
  }

  function togglePlay() {
    playRef.current = !playRef.current;
    setPlaying(playRef.current);
  }

  function restart() {
    psRef.current = initParticles(
      nRef.current,
      rRef.current,
      speed0Ref.current,
    );
  }

  return (
    <div style={styles.wrapper}>
      <div style={{ display: "flex", gap: 12, alignItems: "flex-start" }}>
        <canvas
          ref={simRef}
          width={BOX_W}
          height={BOX_H}
          style={{ ...styles.canvas, display: "block" }}
        />
        <canvas
          ref={histRef}
          width={HC_W}
          height={HC_H}
          style={{ ...styles.canvas, display: "block" }}
        />
      </div>

      <div style={{ ...styles.controls, justifyContent: "center" }}>
        <label style={styles.sliderLabel}>
          <span>
            N = <strong>{nState}</strong>
          </span>
          <input
            type="range"
            min={N_MIN}
            max={N_MAX}
            step={N_STEP}
            value={nState}
            onChange={handleNChange}
            style={styles.slider}
          />
        </label>
        <label style={styles.sliderLabel}>
          <span>
            T = <strong>{tState.toFixed(2)}</strong>
          </span>
          <input
            type="range"
            min={T_MIN}
            max={T_MAX}
            step={T_STEP}
            value={tState}
            onChange={handleTChange}
            style={styles.slider}
          />
        </label>
        <button onClick={togglePlay} style={styles.btn}>
          {playing ? "Pause" : "Play"}
        </button>
        <button onClick={restart} style={styles.btn}>
          Restart
        </button>
      </div>

      <div style={styles.legend}>
        <LegendItem
          color={COLORS.distStroke}
          dash={false}
          label="Simulation data"
        />
        <LegendItem
          color={COLORS.targetStroke}
          dash={false}
          label="Maxwell–Boltzmann"
        />
      </div>
    </div>
  );
}
