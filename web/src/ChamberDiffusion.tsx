import { useRef, useEffect, useState, useCallback } from "react";
import { styles, LegendItem, COLORS } from "./viz-utils";

// ── Layout constants ──────────────────────────────────────────────────────────

const BOX_W = 720;
const BOX_H = 150;

const APERTURE_FRAC = 0.42; // aperture height as a fraction of BOX_H

const DEFAULT_CHAMBERS = 4;
const CHAMBERS_MIN = 2;
const CHAMBERS_MAX = 8;

const DEFAULT_N = 320;
const N_MIN = 80;
const N_MAX = 700;
const N_STEP = 10;

const DEFAULT_SPEED = 1.4;
const SPEED_MIN = 0.4;
const SPEED_MAX = 3.0;
const SPEED_STEP = 0.1;

const HIST_W = BOX_W;
const HIST_H = 170;
const HP = { t: 18, r: 14, b: 34, l: 44 } as const;

// ── Colors ────────────────────────────────────────────────────────────────────

const COL = {
  particle: "rgba(30, 100, 220, 0.78)",
  particleEdge: "rgba(15, 60, 160, 0.90)",
  wall: "#555",
  barFill: "rgba(40, 110, 230, 0.35)",
  barEdge: "rgba(30, 90, 210, 0.82)",
  uniform: "rgba(210, 40, 60, 0.90)",
  axis: "#555",
  text: "#333",
  dim: "#666",
  bg: "#f8f9fb",
} as const;

type Particle = { x: number; y: number; vx: number; vy: number };

// ── Geometry helpers ──────────────────────────────────────────────────────────

function chamberWidth(nChambers: number): number {
  return BOX_W / nChambers;
}

function wallXs(nChambers: number): number[] {
  const xs: number[] = [];
  for (let k = 1; k < nChambers; k++) xs.push(k * chamberWidth(nChambers));
  return xs;
}

function apertureRange(): [number, number] {
  const half = (APERTURE_FRAC * BOX_H) / 2;
  return [BOX_H / 2 - half, BOX_H / 2 + half];
}

// Radius that packs n particles into a chamberW × BOX_H rectangle without overlap.
function packedRadius(n: number, chamberW: number): number {
  const cols = Math.max(1, Math.ceil(Math.sqrt((n * chamberW) / BOX_H)));
  const rows = Math.max(1, Math.ceil(n / cols));
  const cellW = chamberW / cols;
  const cellH = BOX_H / rows;
  return 0.42 * Math.min(cellW, cellH);
}

// ── Initialization ────────────────────────────────────────────────────────────

// Produces a vector of the given length whose angle is chosen uniformly in [π/4, 3π/4, 5π/4, 7π/4]
function randomDiagonalVector(r: number): [number, number] {
  const θ = ((Math.floor(4 * Math.random()) + 0.5) * Math.PI) / 2;
  return [r * Math.cos(θ), r * Math.sin(θ)];
}

function initParticles(
  n: number,
  r: number,
  speed0: number,
  nChambers: number,
): Particle[] {
  const chamberW = chamberWidth(nChambers);
  const cols = Math.ceil(Math.sqrt(n));
  const rows = Math.ceil(n / cols);
  const cw = chamberW / cols;
  const ch = BOX_H / rows;
  const ps: Particle[] = [];
  let k = 0;
  outer: for (let ri = 0; ri < rows; ri++) {
    for (let ci = 0; ci < cols; ci++) {
      if (k++ >= n) break outer;
      const x = Math.min(chamberW - r, Math.max(r, (ci + 0.5) * cw));
      const y = Math.min(BOX_H - r, Math.max(r, (ri + 0.5) * ch));
      const [vx, vy] = randomDiagonalVector(speed0);
      ps.push({ x, y, vx, vy });
    }
  }
  return ps;
}

// ── Physics ───────────────────────────────────────────────────────────────────

function physicsStep(
  ps: Particle[],
  r: number,
  walls: readonly number[],
  gapY0: number,
  gapY1: number,
): void {
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

    // Interior chamber walls: impermeable except through the aperture gap.
    for (const wx of walls) {
      if (p.x <= wx - r || p.x >= wx + r) continue;
      const clearsGap = p.y - r >= gapY0 && p.y + r <= gapY1;
      if (clearsGap) continue;
      if (p.vx >= 0) {
        p.x = 2 * (wx - r) - p.x;
        p.vx = -Math.abs(p.vx);
      } else {
        p.x = 2 * (wx + r) - p.x;
        p.vx = Math.abs(p.vx);
      }
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
  walls: readonly number[],
  gapY0: number,
  gapY1: number,
): void {
  ctx.fillStyle = COL.bg;
  ctx.fillRect(0, 0, BOX_W, BOX_H);

  ctx.strokeStyle = COL.wall;
  ctx.lineWidth = 4;
  for (const wx of walls) {
    ctx.beginPath();
    ctx.moveTo(wx, 0);
    ctx.lineTo(wx, gapY0);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(wx, gapY1);
    ctx.lineTo(wx, BOX_H);
    ctx.stroke();
  }

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

function drawChamberHist(
  ctx: CanvasRenderingContext2D,
  ps: Particle[],
  nChambers: number,
): void {
  const { t, r, b, l } = HP;
  const pw = HIST_W - l - r;
  const ph = HIST_H - t - b;

  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, HIST_W, HIST_H);

  const chamberW = chamberWidth(nChambers);
  const counts = new Array<number>(nChambers).fill(0);
  for (const p of ps) {
    const idx = Math.min(nChambers - 1, Math.max(0, Math.floor(p.x / chamberW)));
    counts[idx]!++;
  }
  const total = Math.max(ps.length, 1);
  const fracs = counts.map((c) => c / total);
  const uniform = 1 / nChambers;

  const yMax = 1.0;
  const cx = (i: number) => l + ((i + 0.5) / nChambers) * pw;
  const barW = (pw / nChambers) * 0.62;
  const y0 = t + ph;

  for (let i = 0; i < nChambers; i++) {
    const bh = (fracs[i]! / yMax) * ph;
    const bx = cx(i) - barW / 2;
    ctx.fillStyle = COL.barFill;
    ctx.fillRect(bx, y0 - bh, barW, bh);
    ctx.strokeStyle = COL.barEdge;
    ctx.lineWidth = 1;
    ctx.strokeRect(bx, y0 - bh, barW, bh);
  }

  // Dashed reference line at the uniform (equilibrium) fraction 1/N.
  const yu = t + (1 - uniform / yMax) * ph;
  ctx.strokeStyle = COL.uniform;
  ctx.lineWidth = 1.6;
  ctx.setLineDash([5, 4]);
  ctx.beginPath();
  ctx.moveTo(l, yu);
  ctx.lineTo(l + pw, yu);
  ctx.stroke();
  ctx.setLineDash([]);

  // Axes
  ctx.strokeStyle = COL.axis;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(l, t);
  ctx.lineTo(l, y0);
  ctx.lineTo(l + pw, y0);
  ctx.stroke();

  ctx.font = "10px system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  ctx.fillStyle = COL.dim;
  for (let i = 0; i < nChambers; i++) {
    ctx.fillText(String(i + 1), cx(i), y0 + 5);
  }
  ctx.fillText("chamber", l + pw / 2, y0 + 18);

  ctx.font = "9px system-ui, sans-serif";
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (let i = 0; i <= 4; i++) {
    const v = (i / 4) * yMax;
    const y = t + (1 - v) * ph;
    ctx.strokeStyle = COL.axis;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(l, y);
    ctx.lineTo(l - 3, y);
    ctx.stroke();
    ctx.fillStyle = COL.dim;
    ctx.fillText(v.toFixed(2), l - 5, y);
  }

  ctx.save();
  ctx.translate(12, t + ph / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.font = "10px system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillStyle = COL.dim;
  ctx.fillText("fraction of particles", 0, 0);
  ctx.restore();

  ctx.font = "bold 11px system-ui, sans-serif";
  ctx.textAlign = "left";
  ctx.textBaseline = "top";
  ctx.fillStyle = COL.text;
  ctx.fillText("occupancy by chamber", l, 2);
}

// ── React component ───────────────────────────────────────────────────────────

export default function ChamberDiffusion() {
  const simRef = useRef<HTMLCanvasElement>(null);
  const histRef = useRef<HTMLCanvasElement>(null);
  const rafRef = useRef<number>(0);
  const playRef = useRef(true);

  // Layout/physics parameters are held in refs so the animation loop always
  // sees current values without needing to restart the rAF loop on slider changes.
  const nChambersRef = useRef(DEFAULT_CHAMBERS);
  const nParticlesRef = useRef(DEFAULT_N);
  const speedRef = useRef(DEFAULT_SPEED);
  const rRef = useRef(packedRadius(DEFAULT_N, chamberWidth(DEFAULT_CHAMBERS)));
  const wallsRef = useRef<number[]>(wallXs(DEFAULT_CHAMBERS));
  const [gapY0Init, gapY1Init] = apertureRange();
  const gapRef = useRef<[number, number]>([gapY0Init, gapY1Init]);
  const psRef = useRef<Particle[]>(
    initParticles(
      nParticlesRef.current,
      rRef.current,
      speedRef.current,
      nChambersRef.current,
    ),
  );

  const [nChambersState, setNChambersState] = useState(DEFAULT_CHAMBERS);
  const [nParticlesState, setNParticlesState] = useState(DEFAULT_N);
  const [speedState, setSpeedState] = useState(DEFAULT_SPEED);
  const [playing, setPlaying] = useState(true);

  const animate = useCallback(() => {
    if (playRef.current) {
      physicsStep(
        psRef.current,
        rRef.current,
        wallsRef.current,
        gapRef.current[0],
        gapRef.current[1],
      );
    }
    const sCtx = simRef.current?.getContext("2d");
    if (sCtx)
      drawSim(
        sCtx,
        psRef.current,
        rRef.current,
        wallsRef.current,
        gapRef.current[0],
        gapRef.current[1],
      );
    const hCtx = histRef.current?.getContext("2d");
    if (hCtx) drawChamberHist(hCtx, psRef.current, nChambersRef.current);
    rafRef.current = requestAnimationFrame(animate);
  }, []);

  useEffect(() => {
    rafRef.current = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(rafRef.current);
  }, [animate]);

  function reinit(nChambers: number, nParticles: number) {
    const r = packedRadius(nParticles, chamberWidth(nChambers));
    nChambersRef.current = nChambers;
    nParticlesRef.current = nParticles;
    rRef.current = r;
    wallsRef.current = wallXs(nChambers);
    psRef.current = initParticles(nParticles, r, speedRef.current, nChambers);
  }

  function handleChambersChange(e: React.ChangeEvent<HTMLInputElement>) {
    const nChambers = parseInt(e.target.value, 10);
    reinit(nChambers, nParticlesRef.current);
    setNChambersState(nChambers);
  }

  function handleNChange(e: React.ChangeEvent<HTMLInputElement>) {
    const n = parseInt(e.target.value, 10);
    reinit(nChambersRef.current, n);
    setNParticlesState(n);
  }

  function handleSpeedChange(e: React.ChangeEvent<HTMLInputElement>) {
    const speedNew = parseFloat(e.target.value);
    const scale = speedNew / speedRef.current;
    for (const p of psRef.current) {
      p.vx *= scale;
      p.vy *= scale;
    }
    speedRef.current = speedNew;
    setSpeedState(speedNew);
  }

  function togglePlay() {
    playRef.current = !playRef.current;
    setPlaying(playRef.current);
  }

  function restart() {
    reinit(nChambersRef.current, nParticlesRef.current);
  }

  return (
    <div style={styles.wrapper}>
      <canvas
        ref={simRef}
        width={BOX_W}
        height={BOX_H}
        style={{ ...styles.canvas, display: "block" }}
      />
      <canvas
        ref={histRef}
        width={HIST_W}
        height={HIST_H}
        style={{ ...styles.canvas, display: "block" }}
      />

      <div style={{ ...styles.controls, justifyContent: "center" }}>
        <label style={styles.sliderLabel}>
          <span>
            chambers = <strong>{nChambersState}</strong>
          </span>
          <input
            type="range"
            min={CHAMBERS_MIN}
            max={CHAMBERS_MAX}
            step={1}
            value={nChambersState}
            onChange={handleChambersChange}
            style={styles.slider}
          />
        </label>
        <label style={styles.sliderLabel}>
          <span>
            N = <strong>{nParticlesState}</strong>
          </span>
          <input
            type="range"
            min={N_MIN}
            max={N_MAX}
            step={N_STEP}
            value={nParticlesState}
            onChange={handleNChange}
            style={styles.slider}
          />
        </label>
        <label style={styles.sliderLabel}>
          <span>
            speed = <strong>{speedState.toFixed(1)}</strong>
          </span>
          <input
            type="range"
            min={SPEED_MIN}
            max={SPEED_MAX}
            step={SPEED_STEP}
            value={speedState}
            onChange={handleSpeedChange}
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
        <LegendItem color={COLORS.distStroke} dash={false} label="Particles" />
        <LegendItem
          color={COLORS.targetStroke}
          dash={true}
          label="Uniform distribution (1/N)"
        />
      </div>
    </div>
  );
}
