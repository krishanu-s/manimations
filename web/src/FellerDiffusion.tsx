/**
 * FellerDiffusion — interactive visualization of the CIR / Feller diffusion process.
 *
 * Evolves an initial rectangular probability distribution on x > 0 according to
 * the Fokker-Planck PDE:
 *
 *   ∂p/∂t = p/λ + (1 + x/λ) ∂p/∂x + x ∂²p/∂x²
 *
 * The long-run steady state is the exponential (Gibbs) distribution g(x) = (1/λ)e^{-x/λ}.
 * Uses an explicit finite-difference scheme; stability requires dt ≤ 0.4 Δx² / x_max.
 *
 * Reference Python implementation: scripts/probability.py, CIRProcess.construct (line 631)
 * and evolve_vals_finite_difference (line 565).
 */

import { useRef, useEffect, useState } from "react";
import {
  makeTransforms,
  drawAxes,
  plotDistribution,
  plotDashedCurve,
  LegendItem,
  styles,
} from "./viz-utils";

// ─── Simulation parameters ───────────────────────────────────────────────────

const XMAX = 5.0;
const NUM_X = 200;
const DEL_X = XMAX / NUM_X;
const XMIN = DEL_X;
const XSPACE: readonly number[] = Object.freeze(
  Array.from({ length: NUM_X }, (_, i) => XMIN + i * DEL_X),
);
// Largest stable timestep (von Neumann-style condition from the Python reference)
const DEL_T = (0.4 * DEL_X * DEL_X) / XMAX;
// Substeps per animation frame — controls how fast the sim runs relative to real time
const SUBSTEPS = 80;
const INIT_HALF_WIDTH = 0.2;

const YMAX = 8.0;

// ─── Simulation logic ────────────────────────────────────────────────────────

function makeInitial(mean: number): Float64Array {
  return Float64Array.from(XSPACE, (x) =>
    Math.abs(x - mean) < INIT_HALF_WIDTH ? 1 / (2 * INIT_HALF_WIDTH) : 0.0,
  );
}

/**
 * Advances the distribution by SUBSTEPS fine timesteps in-place,
 * reusing a pre-allocated dp scratch buffer.
 */
function evolve(
  mean: number,
  vals: Float64Array,
  dp: Float64Array,
): Float64Array {
  const n = vals.length;

  for (let s = 0; s < SUBSTEPS; s++) {
    // Neumann-style boundary at x=0
    dp[0] = vals[0]! / mean + (vals[1]! - vals[0]!) / DEL_X;
    // Zero-flux boundary at x_max
    dp[n - 1] = 0.0;

    for (let i = 1; i < n - 1; i++) {
      const x = XSPACE[i]!;
      const p = vals[i]!;
      const dv = (vals[i + 1]! - vals[i - 1]!) / (2 * DEL_X);
      const d2v = (vals[i + 1]! + vals[i - 1]! - 2 * p) / (DEL_X * DEL_X);
      dp[i] = p / mean + (1 + x / mean) * dv + x * d2v;
    }

    let sum = 0;
    for (let i = 0; i < n; i++) {
      const v = vals[i]! + DEL_T * dp[i]!;
      vals[i] = v < 0 ? 0 : v;
      sum += vals[i]!;
    }
    const norm = DEL_X * sum;
    if (norm > 0) {
      for (let i = 0; i < n; i++) vals[i] = vals[i]! / norm;
    }
  }

  return vals;
}

// ─── Rendering ───────────────────────────────────────────────────────────────

function renderFrame(
  ctx: CanvasRenderingContext2D,
  vals: Float64Array,
  mean: number,
  W: number,
  H: number,
): void {
  ctx.clearRect(0, 0, W, H);

  // Use x origin at 0 (not XMIN=DEL_X) so the y-axis sits flush at the left edge
  const tr = makeTransforms(0, XMAX, YMAX, W, H);

  drawAxes(ctx, 0, XMAX, YMAX, 1, 2, tr);
  plotDashedCurve(ctx, XSPACE, (x) => Math.exp(-x / mean) / mean, tr);
  plotDistribution(ctx, XSPACE, vals, tr);
}

// ─── React component ─────────────────────────────────────────────────────────

interface SimState {
  vals: Float64Array;
  dp: Float64Array;
  mean: number;
  paused: boolean;
  rafId: number;
}

interface Props {
  canvasWidth?: number;
  canvasHeight?: number;
}

export default function FellerDiffusion({
  canvasWidth = 660,
  canvasHeight = 400,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const sim = useRef<SimState>({
    vals: makeInitial(2.0),
    dp: new Float64Array(NUM_X),
    mean: 2.0,
    paused: true,
    rafId: 0,
  });

  const [paused, setPaused] = useState(true);
  const [mean, setMean] = useState(2.0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const W = canvas.width;
    const H = canvas.height;

    function frame() {
      const s = sim.current;
      if (!s.paused) evolve(s.mean, s.vals, s.dp);
      renderFrame(ctx!, s.vals, s.mean, W, H);
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
    sim.current.vals = makeInitial(sim.current.mean);
  };

  const handleMeanChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const v = parseFloat(e.target.value);
    sim.current.mean = v;
    setMean(v);
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
        <label style={styles.sliderLabel}>
          <span>
            Mean &lambda;&nbsp;=&nbsp;<strong>{mean.toFixed(2)}</strong>
          </span>
          <input
            type="range"
            min="0.3"
            max="4.0"
            step="0.05"
            value={mean}
            onChange={handleMeanChange}
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
          label={`Target: (1/λ) e^{−x/λ},  λ = ${mean.toFixed(2)}`}
        />
      </div>
    </div>
  );
}
