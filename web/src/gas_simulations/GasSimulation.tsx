import { useRef, useEffect, useState, useCallback } from "react";
import { styles, LegendItem, COLORS } from "../viz-utils";
import { SimSettings, Particle, physicsStepBox } from "./physics";
import { COL, randomDiagonalVector, randomRadialVector } from "./utils";

// ── Layout constants ──────────────────────────────────────────────────────────

// Total region allotted for physics simulation
const BOX_W = 400;
const BOX_H = 500;

// Note that temperature T means initial speed is √(T).
const DEFAULT_T = 1;
const T_MIN = 0.05;
const T_MAX = 4;
const T_STEP = 0.05;

// Timestep Δt is normalized to 1. Since
// - the velocity value here represents vΔt and
// - the gravity value here represents g(Δt)^2
// it follows that gravity values should be small.
const DEFAULT_G = 0.0;
const G_MIN = 0.0;
const G_MAX = 0.03;
const G_STEP = 0.001;

// Total region allotted for histograms.
const HC_W = 280;
const HC_H = 500;

// Individual histograms
const HIST_H = 115;
const N_HIST = 4;
const HIST_OFFSET = 125;
const H_YOFFS = [
  5,
  5 + HIST_OFFSET,
  5 + 2 * HIST_OFFSET,
  5 + 3 * HIST_OFFSET,
] as const;
const HP = { t: 20, r: 14, b: 32, l: 30 } as const;
const N_BINS = 60;

// Number of chambers
const DEFAULT_LEVELS = 4;
const LEVELS_MIN = 2;
const LEVELS_MAX = 16;

// Number of particles
const DEFAULT_N = 400;
const N_MIN = 60;
const N_MAX = 1000;
const N_STEP = 10;

function radiusFor(n: number) {
  return 40 / Math.sqrt(n);
}

// ── Initialization ────────────────────────────────────────────────────────────

// Initialize particles uniformly in the box
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

// ── Renderers ─────────────────────────────────────────────────────────────────

function drawSim(
  ctx: CanvasRenderingContext2D,
  { ps, r, g, boxW, boxH }: SimSettings,
  nLevels: number,
): void {
  // Rectangular box
  ctx.fillStyle = COL.bg;
  ctx.fillRect(0, 0, boxW, boxH);

  // Virtual dividers between regions and region labels
  ctx.strokeStyle = COL.vdiv;
  ctx.lineWidth = 1;
  for (let i = 1; i < nLevels; i++) {
    ctx.beginPath();
    ctx.moveTo(0, (i / nLevels) * boxH);
    ctx.lineTo(boxW, (i / nLevels) * boxH);
    ctx.stroke();
  }

  ctx.fillStyle = COL.text;
  ctx.strokeStyle = COL.dim;
  ctx.font = "14px system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  for (let i = 1; i < nLevels + 1; i++) {
    ctx.fillText(i.toString(), BOX_W / 2, (BOX_H * (i - 0.5)) / nLevels - 8);
  }

  // Individual particles
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
  nBins: number,
): void {
  const { t, r, b, l } = HP;
  const pw = HC_W - l - r;
  const ph = HIST_H - t - b;
  const range = xMax - xMin;
  const bw = range / nBins;

  ctx.fillStyle = "#fff";
  ctx.fillRect(0, yOff, HC_W, HIST_H);

  const counts = new Array<number>(nBins).fill(0);
  for (const v of values) {
    const idx = Math.floor(((v - xMin) / range) * nBins);
    if (idx >= 0 && idx < nBins) counts[idx]!++;
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

  // Bars
  const bpx = pw / nBins;
  for (let i = 0; i < nBins; i++) {
    const bx = cx(xMin + i * bw);
    const bh = (dens[i]! / yMax) * ph;
    ctx.fillStyle = COL.barFill;
    ctx.fillRect(bx, y0 - bh, bpx - 0.5, bh);
    ctx.strokeStyle = COL.barEdge;
    ctx.lineWidth = 0.5;
    ctx.strokeRect(bx, y0 - bh, bpx - 0.5, bh);
  }

  // Theory curve
  ctx.beginPath();
  ctx.strokeStyle = COL.theory;
  ctx.lineWidth = 1.8;
  for (let i = 0; i <= 200; i++) {
    const x = xMin + (i / 200) * range;
    if (i === 0) ctx.moveTo(cx(x), cy(theoryFn(x)));
    else ctx.lineTo(cx(x), cy(theoryFn(x)));
  }
  ctx.stroke();

  // Axis
  ctx.strokeStyle = COL.axis;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(l, yOff + t);
  ctx.lineTo(l, y0);
  ctx.lineTo(l + pw, y0);
  ctx.stroke();

  // Text
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

// Draw the histograms. TODO Toggles
function drawHistograms(
  ctx: CanvasRenderingContext2D,
  ps: Particle[],
  g: number,
  nLevels: number,
): void {
  ctx.fillStyle = COL.bg;
  ctx.fillRect(0, 0, HC_W, HC_H);

  const vxArr = ps.map((p) => p.vx);
  const vyArr = ps.map((p) => p.vy);
  const keArr = ps.map((p) => 0.5 * (p.vx * p.vx + p.vy * p.vy));
  const levelsArr = ps.map((p) => nLevels * (1 - p.y / BOX_H));

  const kT = keArr.reduce((s, k) => s + k, 0) / ps.length;
  const sigma = Math.sqrt(Math.max(kT, 1e-9));

  // Fix the x-scale of the vx and vy plots
  // const vRange = Math.max(4 * sigma, 0.1);
  const vRange = 4.0;

  // Fix the x-scale of the kinetic energy plot
  // const keMax = Math.max(5 * kT, 0.1);
  const keMax = 5.0;

  // Theoretical distribution for velocity
  const gaussFn = (v: number) =>
    (1 / (Math.sqrt(2 * Math.PI) * sigma)) * Math.exp((-v * v) / (2 * kT));

  // Theoretical distribution for kinetic energy
  const expFnKE = (ke: number) => (ke >= 0 ? (1 / kT) * Math.exp(-ke / kT) : 0);

  // Theoretical distribution for y-position
  // TODO Have to figure out the constants here. Should be a function that the integral from 0 to nLevels is equal to 1,
  // and such that it is exponentially decreasing
  const mean = (BOX_H / nLevels) * g;
  const C = mean / (1 - Math.exp(-nLevels));
  const expFnPE = (pe: number) =>
    pe >= 0 ? (g > 0 ? C * Math.exp(-pe * mean) : 1 / nLevels) : 0;

  // Position y-component
  drawOneHist(
    ctx,
    H_YOFFS[0],
    levelsArr,
    0,
    nLevels,
    "y position",
    "y",
    expFnPE,
    nLevels,
  );

  // Kinetic energy
  drawOneHist(
    ctx,
    H_YOFFS[1],
    keArr,
    0,
    keMax,
    "kinetic energy",
    "KE",
    expFnKE,
    N_BINS,
  );

  // Velocity x-component
  drawOneHist(
    ctx,
    H_YOFFS[2],
    vxArr,
    -vRange,
    vRange,
    "vx velocities",
    "vx",
    gaussFn,
    N_BINS,
  );

  // Velocity y-component
  drawOneHist(
    ctx,
    H_YOFFS[3],
    vyArr,
    -vRange,
    vRange,
    "vy velocities",
    "vy",
    gaussFn,
    N_BINS,
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
  const gRef = useRef(DEFAULT_G);
  const speed0Ref = useRef(Math.sqrt(DEFAULT_T));
  const psRef = useRef<Particle[]>(
    initParticles(nRef.current, rRef.current, speed0Ref.current),
  );
  const nLevelsRef = useRef(DEFAULT_LEVELS);

  const [nState, setNState] = useState(DEFAULT_N);
  const [tState, setTState] = useState(DEFAULT_T);
  const [gState, setGState] = useState(DEFAULT_G);
  const [playing, setPlaying] = useState(true);
  const [nLevelsState, setNLevelsState] = useState(DEFAULT_LEVELS);

  const animate = useCallback(() => {
    if (playRef.current)
      physicsStepBox({
        ps: psRef.current,
        r: rRef.current,
        g: gRef.current,
        boxW: BOX_W,
        boxH: BOX_H,
      });
    const sCtx = simRef.current?.getContext("2d");
    if (sCtx)
      drawSim(
        sCtx,
        {
          ps: psRef.current,
          r: rRef.current,
          g: 0,
          boxW: BOX_W,
          boxH: BOX_H,
        },
        nLevelsRef.current,
      );
    const hCtx = histRef.current?.getContext("2d");
    if (hCtx)
      drawHistograms(hCtx, psRef.current, gRef.current, nLevelsRef.current);
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

  function handleLevelsChange(e: React.ChangeEvent<HTMLInputElement>) {
    const nLevelsNew = parseInt(e.target.value);
    nLevelsRef.current = nLevelsNew;
    setNLevelsState(nLevelsNew);
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
    psRef.current = initParticles(
      nRef.current,
      rRef.current,
      speed0Ref.current,
    );
  }

  return (
    <div style={styles.wrapper}>
      <div style={{ display: "flex", gap: 12, alignItems: "flex-start" }}>
        {/*Display simulation*/}
        <canvas
          ref={simRef}
          width={BOX_W}
          height={BOX_H}
          style={{ ...styles.canvas, display: "block" }}
        />
        {/*Display histograms*/}
        <canvas
          ref={histRef}
          width={HC_W}
          height={HC_H}
          style={{ ...styles.canvas, display: "block" }}
        />
      </div>

      {/*Sliders*/}
      <div style={{ ...styles.controls, justifyContent: "center" }}>
        {/*Number of particles*/}
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

        {/*Number of Levels*/}
        <label style={styles.sliderLabel}>
          <span>
            levels = <strong>{nLevelsState}</strong>
          </span>
          <input
            type="range"
            min={LEVELS_MIN}
            max={LEVELS_MAX}
            step={1}
            value={nLevelsState}
            onChange={handleLevelsChange}
            style={styles.slider}
          />
        </label>

        {/*Temperature*/}
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
