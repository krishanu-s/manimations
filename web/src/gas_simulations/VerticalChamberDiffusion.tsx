// Box divided into chambers arranged vertically, with apertures between them
// At the right is shown a histogram of the ratio of particles in each chamber, along
// with sliders to control the gravitational constant
// An option is to exactly the same backend as ChamberDiffusion but with (g.x, g.y) = (g, 0),
// and then draw the same simulation in a reflected manner.

import { useRef, useEffect, useState, useCallback } from "react";
import { styles, LegendItem, COLORS } from "../viz-utils";
import {
  Particle,
  physicsStepChambers,
  physicsStepChambersVertical,
} from "./physics";
import { COL, randomDiagonalVector, randomRadialVector } from "./utils";

// ── Layout constants ──────────────────────────────────────────────────────────

const BOX_W = 240;
const BOX_H = 720;

const APERTURE_FRAC = 0.7; // aperture height as a fraction of BOX_H

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

// Timestep Δt is normalized to 1. Since
// - the velocity value here represents vΔt and
// - the gravity value here represents g(Δt)^2
// it follows that gravity values should be small.
const DEFAULT_G = 0.0;
const G_MIN = 0.0;
const G_MAX = 0.05;
const G_STEP = 0.001;

const HIST_W = BOX_W;
const HIST_H = 170;
const HP = { t: 18, r: 14, b: 34, l: 44 } as const;

// ── Geometry helpers ──────────────────────────────────────────────────────────

function chamberHeight(nChambers: number): number {
  return BOX_H / nChambers;
}

function wallYs(nChambers: number): number[] {
  const ys: number[] = [];
  for (let k = 1; k < nChambers; k++) ys.push(k * chamberHeight(nChambers));
  return ys;
}

function apertureRange(): [number, number] {
  const half = (APERTURE_FRAC * BOX_W) / 2;
  return [BOX_W / 2 - half, BOX_W / 2 + half];
}

// Radius that packs n particles into a chamberH × BOX_W rectangle without overlap.
function packedRadius(n: number, chamberH: number): number {
  const cols = Math.max(1, Math.ceil(Math.sqrt((n * 0.25 * chamberH) / BOX_W)));
  const rows = Math.max(1, Math.ceil(n / cols));
  const cellW = (0.25 * chamberH) / cols;
  const cellH = BOX_W / rows;
  return 0.15 * Math.min(cellW, cellH);
}

// ── Initialization ────────────────────────────────────────────────────────────

function initParticles(
  n: number,
  r: number,
  speed0: number,
  nChambers: number,
): Particle[] {
  const chamberH = chamberHeight(nChambers);
  const cols = Math.ceil(Math.sqrt(n));
  const rows = Math.ceil(n / cols);
  const ch = chamberH / cols;
  const cw = BOX_W / rows;
  const ps: Particle[] = [];
  let k = 0;
  outer: for (let ri = 0; ri < rows; ri++) {
    for (let ci = 0; ci < cols; ci++) {
      if (k++ >= n) break outer;
      const y = Math.min(chamberH - r, Math.max(r, (ci + 0.5) * ch));
      const x = Math.min(BOX_W - r, Math.max(r, (ri + 0.5) * cw));
      const [vx, vy] = randomDiagonalVector(speed0);
      ps.push({ x, y, vx, vy });
    }
  }
  return ps;
}

// ── Renderers ─────────────────────────────────────────────────────────────────

function drawSim(
  ctx: CanvasRenderingContext2D,
  ps: Particle[],
  r: number,
  walls: readonly number[],
  gapX0: number,
  gapX1: number,
): void {
  ctx.fillStyle = COL.bg;
  ctx.fillRect(0, 0, BOX_W, BOX_H);

  // Draw apertures
  ctx.strokeStyle = COL.wall;
  ctx.lineWidth = 4;
  for (const wy of walls) {
    ctx.beginPath();
    ctx.moveTo(0, wy);
    ctx.lineTo(gapX0, wy);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(gapX1, wy);
    ctx.lineTo(BOX_W, wy);
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

  const chamberH = chamberHeight(nChambers);
  const counts = new Array<number>(nChambers).fill(0);
  for (const p of ps) {
    const idx = Math.min(
      nChambers - 1,
      Math.max(0, Math.floor(p.y / chamberH)),
    );
    counts[nChambers - 1 - idx]!++;
  }
  const total = Math.max(ps.length, 1);
  const fracs = counts.map((c) => c / total);
  const uniform = 1 / nChambers;

  const yMax = 1.0;
  const cx = (i: number) => l + ((i + 0.5) / nChambers) * pw;
  const barW = (pw / nChambers) * 1.0;
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

  // Dashed reference line according to an exponential decay curve
  const yu = t + (1 - uniform / yMax) * ph;
  ctx.strokeStyle = COL.theory;
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

export default function VerticalChamberDiffusion() {
  const simRef = useRef<HTMLCanvasElement>(null);
  const histRef = useRef<HTMLCanvasElement>(null);
  const rafRef = useRef<number>(0);
  const playRef = useRef(true);

  // Layout/physics parameters are held in refs so the animation loop always
  // sees current values without needing to restart the rAF loop on slider changes.
  const nChambersRef = useRef(DEFAULT_CHAMBERS);
  const nParticlesRef = useRef(DEFAULT_N);
  const [gState, setGState] = useState(DEFAULT_G);
  const speedRef = useRef(DEFAULT_SPEED);
  const gRef = useRef(DEFAULT_G);
  const rRef = useRef(packedRadius(DEFAULT_N, chamberHeight(DEFAULT_CHAMBERS)));
  const wallsRef = useRef<number[]>(wallYs(DEFAULT_CHAMBERS));
  const [gapX0Init, gapX1Init] = apertureRange();
  const gapRef = useRef<[number, number]>([gapX0Init, gapX1Init]);
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
      physicsStepChambersVertical(
        {
          ps: psRef.current,
          r: rRef.current,
          g: gRef.current,
          boxW: BOX_W,
          boxH: BOX_H,
        },
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
    const r = packedRadius(nParticles, chamberHeight(nChambers));
    nChambersRef.current = nChambers;
    nParticlesRef.current = nParticles;
    rRef.current = r;
    wallsRef.current = wallYs(nChambers);
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

  function handleGChange(e: React.ChangeEvent<HTMLInputElement>) {
    const gNew = parseFloat(e.target.value);
    gRef.current = gNew;
    setGState(gNew);
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
      <div style={{ display: "flex", gap: 12, alignItems: "flex-start" }}>
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
      </div>

      <div style={{ ...styles.controls, justifyContent: "center" }}>
        {/*Number of chambers*/}
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

        {/*Number of particles*/}
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

        {/*Temperature*/}
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
        {/*Gravity*/}
        <label style={styles.sliderLabel}>
          <span>
            g = <strong>{(100 * gState).toFixed(2)}</strong>
          </span>
          <input
            type="range"
            min={G_MIN}
            max={G_MAX}
            step={G_STEP}
            value={gState}
            onChange={handleGChange}
            style={styles.slider}
          />
        </label>

        {/*Buttons*/}
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
