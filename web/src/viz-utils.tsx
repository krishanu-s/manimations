/**
 * viz-utils.tsx — shared canvas drawing primitives and React UI components
 * used by FellerDiffusion and OrnsteinUhlenbeck (and any future visualizations).
 */

// ─── Layout ──────────────────────────────────────────────────────────────────

export const PAD = { top: 20, right: 24, bottom: 44, left: 54 } as const;

// ─── Colors ───────────────────────────────────────────────────────────────────

export const COLORS = {
  distFill:   "rgba(40, 110, 230, 0.18)",
  distStroke: "rgba(30, 90, 210, 0.9)",
  targetStroke: "rgba(210, 40, 60, 0.85)",
  axis:  "#555",
  label: "#333",
  tick:  "#444",
  grid:  "rgba(0,0,0,0.055)",
} as const;

// ─── Coordinate transforms ────────────────────────────────────────────────────

export interface Transforms {
  cx: (x: number) => number;
  cy: (y: number) => number;
}

/**
 * Returns canvas coordinate functions for a plot region described by
 * [xMin, xMax] × [0, yMax], inset by PAD.
 */
export function makeTransforms(
  xMin: number,
  xMax: number,
  yMax: number,
  W: number,
  H: number,
): Transforms {
  const pw = W - PAD.left - PAD.right;
  const ph = H - PAD.top - PAD.bottom;
  return {
    cx: (x) => PAD.left + ((x - xMin) / (xMax - xMin)) * pw,
    cy: (y) => H - PAD.bottom - (Math.min(Math.max(y, 0), yMax) / yMax) * ph,
  };
}

// ─── Axes ────────────────────────────────────────────────────────────────────

/**
 * Draws a complete set of axes, tick marks, grid lines, and "x" / "p(x)" labels.
 *
 * @param xMin      Left edge of the data domain (origin of the x-axis may differ)
 * @param xMax      Right edge of the data domain
 * @param yMax      Top of the y-axis
 * @param xTickStep Spacing between x ticks (integer multiples drawn)
 * @param yTickStep Spacing between y ticks (integer multiples drawn)
 * @param tr        Coordinate transforms from makeTransforms
 */
export function drawAxes(
  ctx: CanvasRenderingContext2D,
  xMin: number,
  xMax: number,
  yMax: number,
  xTickStep: number,
  yTickStep: number,
  tr: Transforms,
): void {
  const { cx, cy } = tr;

  // Faint grid lines (drawn first so axes sit on top)
  ctx.lineWidth = 1;
  ctx.strokeStyle = COLORS.grid;
  for (let xi = Math.ceil(xMin / xTickStep) * xTickStep; xi <= xMax + 1e-9; xi += xTickStep) {
    if (xi === 0) continue;
    ctx.beginPath();
    ctx.moveTo(cx(xi), cy(0));
    ctx.lineTo(cx(xi), cy(yMax));
    ctx.stroke();
  }
  for (let yi = yTickStep; yi <= yMax + 1e-9; yi += yTickStep) {
    ctx.beginPath();
    ctx.moveTo(cx(xMin), cy(yi));
    ctx.lineTo(cx(xMax), cy(yi));
    ctx.stroke();
  }

  // Axis lines
  ctx.strokeStyle = COLORS.axis;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(cx(xMin), cy(0));
  ctx.lineTo(cx(xMax), cy(0));
  ctx.moveTo(cx(0), cy(0));
  ctx.lineTo(cx(0), cy(yMax));
  ctx.stroke();

  // x-axis tick marks and labels (skip x = 0 — the y-axis marks it)
  ctx.fillStyle = COLORS.tick;
  ctx.font = "13px system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  ctx.lineWidth = 1;
  for (let xi = Math.ceil(xMin / xTickStep) * xTickStep; xi <= xMax + 1e-9; xi += xTickStep) {
    ctx.strokeStyle = COLORS.axis;
    ctx.beginPath();
    ctx.moveTo(cx(xi), cy(0));
    ctx.lineTo(cx(xi), cy(0) + 5);
    ctx.stroke();
    if (xi !== 0) ctx.fillText(String(xi), cx(xi), cy(0) + 7);
  }

  // y-axis tick marks and labels (skip y = 0)
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (let yi = yTickStep; yi <= yMax + 1e-9; yi += yTickStep) {
    ctx.strokeStyle = COLORS.axis;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(cx(0), cy(yi));
    ctx.lineTo(cx(0) - 5, cy(yi));
    ctx.stroke();
    ctx.fillText(String(yi), cx(0) - 8, cy(yi));
  }

  // Axis name labels
  ctx.fillStyle = COLORS.label;
  ctx.font = "14px system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "alphabetic";
  ctx.fillText("x", cx(xMax) + 14, cy(0) + 4);
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  ctx.fillText("p(x)", cx(0), PAD.top - 14);
}

// ─── Curve drawing ────────────────────────────────────────────────────────────

/**
 * Fills and strokes a probability distribution as a closed polygon sitting on y = 0.
 */
export function plotDistribution(
  ctx: CanvasRenderingContext2D,
  xspace: readonly number[],
  vals: Float64Array,
  tr: Transforms,
): void {
  const { cx, cy } = tr;
  ctx.fillStyle = COLORS.distFill;
  ctx.strokeStyle = COLORS.distStroke;
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  ctx.moveTo(cx(xspace[0]!), cy(0));
  for (let i = 0; i < xspace.length; i++) {
    ctx.lineTo(cx(xspace[i]!), cy(vals[i]!));
  }
  ctx.lineTo(cx(xspace[xspace.length - 1]!), cy(0));
  ctx.closePath();
  ctx.fill();
  ctx.stroke();
}

/**
 * Strokes a target / reference curve as a dashed red line.
 *
 * @param fn  Function evaluated at each point in xspace
 */
export function plotDashedCurve(
  ctx: CanvasRenderingContext2D,
  xspace: readonly number[],
  fn: (x: number) => number,
  tr: Transforms,
): void {
  const { cx, cy } = tr;
  ctx.strokeStyle = COLORS.targetStroke;
  ctx.lineWidth = 2;
  ctx.setLineDash([7, 5]);
  ctx.beginPath();
  for (let i = 0; i < xspace.length; i++) {
    const x = xspace[i]!;
    if (i === 0) ctx.moveTo(cx(x), cy(fn(x)));
    else ctx.lineTo(cx(x), cy(fn(x)));
  }
  ctx.stroke();
  ctx.setLineDash([]);
}

// ─── React components ─────────────────────────────────────────────────────────

export function LegendItem({
  color,
  dash,
  label,
}: {
  color: string;
  dash: boolean;
  label: string;
}) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
      <svg width="32" height="12" style={{ flexShrink: 0 }}>
        <line
          x1="0"
          y1="6"
          x2="32"
          y2="6"
          stroke={color}
          strokeWidth="2.5"
          strokeDasharray={dash ? "6 4" : undefined}
        />
      </svg>
      <span>{label}</span>
    </div>
  );
}

export const styles = {
  wrapper: {
    display: "inline-flex",
    flexDirection: "column" as const,
    gap: "12px",
    fontFamily: "system-ui, sans-serif",
    fontSize: "14px",
    color: "#222",
    padding: "16px",
    background: "#fafafa",
    border: "1px solid #ddd",
    borderRadius: "8px",
  },
  canvas: {
    display: "block",
    background: "#fff",
    border: "1px solid #e0e0e0",
    borderRadius: "4px",
  },
  controls: {
    display: "flex",
    alignItems: "center",
    gap: "12px",
    flexWrap: "wrap" as const,
  },
  btn: {
    padding: "6px 14px",
    border: "1px solid #bbb",
    borderRadius: "4px",
    background: "#fff",
    cursor: "pointer",
    fontSize: "13px",
    lineHeight: 1.4,
  },
  sliderLabel: {
    display: "flex",
    alignItems: "center",
    gap: "10px",
  },
  slider: {
    width: "180px",
    accentColor: "#1e5ad2",
  },
  legend: {
    display: "flex",
    gap: "20px",
    fontSize: "13px",
    color: "#444",
    flexWrap: "wrap" as const,
  },
} as const;
