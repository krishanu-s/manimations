/**
 * OrnsteinUhlenbeck.tsx — interactive visualization of the Ornstein–Uhlenbeck process.
 *
 * Evolves an initial bimodal distribution toward the Gaussian N(μ, σ²) with the
 * same mean and variance. The OU evolution is:
 *
 *   X^(t) = e^{−t}(X₀ − μ) + μ + √(1−e^{−2t})·σ·G,   G ~ N(0,1)
 *
 * Starting from two symmetric Gaussian bumps at μ ± d (with ε² + d² = σ²),
 * the distribution at time t is computable in closed form — no time-stepping needed:
 *
 *   p_t(x) = ½ N(μ − d·e^{−t}, σ_t)(x)  +  ½ N(μ + d·e^{−t}, σ_t)(x)
 *
 * where σ_t = √(ε²·e^{−2t} + σ²·(1−e^{−2t})).
 *
 * Reference Python implementation: scripts/probability.py, OUProcess (line 350).
 */

import { useRef, useEffect, useState } from "react";
import {
  makeTransforms,
  drawAxes,
  plotDistribution,
  plotDashedCurve,
  LegendItem,
  styles,
  PAD,
} from "./viz-utils";

// ─── Grid ─────────────────────────────────────────────────────────────────────
const XMIN = -6.0;
const XMAX = 6.0;
const NUM_X = 300;
const DEL_X = (XMAX - XMIN) / (NUM_X - 1);
const XSPACE: readonly number[] = Object.freeze(
  Array.from({ length: NUM_X }, (_, i) => XMIN + i * DEL_X),
);

const DT_PER_FRAME = 0.001;
const T_MAX = 8.0;
const YMAX = 4.0;

// ─── Analytics ───────────────────────────────────────────────────────────────

function gaussianPDF(mean: number, std: number, x: number): number {
  const z = (x - mean) / std;
  return Math.exp(-0.5 * z * z) / (std * Math.sqrt(2 * Math.PI));
}

/**
 * Computes p_t analytically.
 *
 * The initial distribution is two equal Gaussian bumps at μ ± d with std ε,
 * where d² + ε² = σ² so that mean = μ and variance = σ² are preserved for all t.
 * ε is kept small so the bumps are visually distinct at t = 0.
 */
function computeDistribution(
  t: number,
  mean: number,
  std: number,
): Float64Array {
  const epsilon = Math.min(0.15, std / 3);
  const d = Math.sqrt(Math.max(0, std * std - epsilon * epsilon));
  const et = Math.exp(-t);
  const sigmaT = Math.sqrt(
    epsilon * epsilon * et * et + std * std * (1 - et * et),
  );
  const offset = d * et;

  const vals = new Float64Array(NUM_X);
  for (let i = 0; i < NUM_X; i++) {
    const x = XSPACE[i]!;
    vals[i] =
      0.5 * gaussianPDF(mean - offset, sigmaT, x) +
      0.5 * gaussianPDF(mean + offset, sigmaT, x);
  }
  return vals;
}

/**
 * Computes the entropy of a distribution.
 */
function computeEntropy(vals: Float64Array): number {
  let sum = 0.0;
  for (let i = 0; i < NUM_X; i++) {
    const p = vals[i]!;
    if (p > 0) sum += p * Math.log(p);
  }
  return -sum * DEL_X;
}

// ─── Rendering ───────────────────────────────────────────────────────────────

function renderFrame(
  ctx: CanvasRenderingContext2D,
  vals: Float64Array,
  mean: number,
  std: number,
  t: number,
  W: number,
  H: number,
): void {
  ctx.clearRect(0, 0, W, H);

  const tr = makeTransforms(XMIN, XMAX, YMAX, W, H);

  drawAxes(ctx, XMIN, XMAX, YMAX, 1, 1, tr);
  plotDashedCurve(ctx, XSPACE, (x) => gaussianPDF(mean, std, x), tr);
  plotDistribution(ctx, XSPACE, vals, tr);

  // Simulation time label (unique to this component)
  ctx.fillStyle = "rgba(80, 80, 80, 0.8)";
  ctx.font = "12px system-ui, sans-serif";
  ctx.textAlign = "right";
  ctx.textBaseline = "top";
  ctx.fillText(`t = ${t.toFixed(2)}`, W - PAD.right - 4, PAD.top + 2);
  // Probably distribution entropy value
  const hX = computeEntropy(vals);
  ctx.fillText(`h(X) = ${hX.toFixed(2)}`, W - PAD.right - 4, PAD.top + 2 + 20);
}

// ─── React component ─────────────────────────────────────────────────────────

interface SimState {
  t: number;
  mean: number;
  std: number;
  paused: boolean;
  rafId: number;
}

interface Props {
  canvasWidth?: number;
  canvasHeight?: number;
}

export default function OrnsteinUhlenbeck({
  canvasWidth = 660,
  canvasHeight = 400,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const sim = useRef<SimState>({
    t: 0,
    mean: 0.0,
    std: 1.0,
    paused: true,
    rafId: 0,
  });

  const [paused, setPaused] = useState(true);
  const [mean, setMean] = useState(0.0);
  const [std, setStd] = useState(1.0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const W = canvas.width;
    const H = canvas.height;

    function frame() {
      const s = sim.current;
      if (!s.paused) {
        s.t = Math.min(s.t + DT_PER_FRAME, T_MAX);
      }
      const vals = computeDistribution(s.t, s.mean, s.std);
      renderFrame(ctx!, vals, s.mean, s.std, s.t, W, H);
      s.rafId = requestAnimationFrame(frame);
    }

    sim.current.rafId = requestAnimationFrame(frame);
    return () => cancelAnimationFrame(sim.current.rafId);
  }, []);

  const togglePause = () => {
    sim.current.paused = !sim.current.paused;
    setPaused(sim.current.paused);
  };

  const reset = () => {
    sim.current.t = 0;
  };

  const handleMeanChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const v = parseFloat(e.target.value);
    sim.current.mean = v;
    sim.current.t = 0;
    setMean(v);
  };

  const handleStdChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const v = parseFloat(e.target.value);
    sim.current.std = v;
    sim.current.t = 0;
    setStd(v);
  };

  return (
    <div style={styles.wrapper}>
      <canvas
        ref={canvasRef}
        width={canvasWidth}
        height={canvasHeight}
        style={styles.canvas}
      />

      <div style={styles.controls}>
        <button onClick={togglePause} style={styles.btn}>
          {paused ? "▶ Resume" : "⏸ Pause"}
        </button>
        <button onClick={reset} style={styles.btn}>
          ↺ Reset
        </button>
        {/*<label style={styles.sliderLabel}>
          <span>
            Mean &mu;&nbsp;=&nbsp;<strong>{mean.toFixed(1)}</strong>
          </span>
          <input
            type="range"
            min="-2"
            max="2"
            step="0.1"
            value={mean}
            onChange={handleMeanChange}
            style={styles.slider}
          />
        </label>*/}
        <label style={styles.sliderLabel}>
          <span>
            Std &sigma;&nbsp;=&nbsp;<strong>{std.toFixed(2)}</strong>
          </span>
          <input
            type="range"
            min="0.3"
            max="2.0"
            step="0.05"
            value={std}
            onChange={handleStdChange}
            style={styles.slider}
          />
        </label>
      </div>

      <div style={styles.legend}>
        <LegendItem
          color="rgba(30,90,210,0.9)"
          dash={false}
          label="Current distribution p(x, t)"
        />
        <LegendItem
          color="rgba(210,40,60,0.85)"
          dash={true}
          label={`Target: N(μ = ${mean.toFixed(1)}, σ² = ${(std * std).toFixed(2)})`}
        />
      </div>
    </div>
  );
}
