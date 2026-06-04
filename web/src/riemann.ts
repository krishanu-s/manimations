/**
 * Riemann Mapping Theorem — interactive animation.
 *
 * Left panel: a Fourier-series curve with interior Cartesian grid lines
 * (static reference).  Right panel: unit circle.  The slider drives the
 * Stein-Shakarchi iteration that conformally maps the domain onto the unit
 * disk, showing the grid homotoping from its original shape to filling the
 * disk.  Each step applies:
 *   z → φ_{√α}( √( φ_α(z) ) ) · e^{i·arg(α)/2}
 * where φ_a is the Poincaré disk automorphism interchanging 0 and a, and √
 * is the branch of the square root continuous on the image of the domain.
 */

const TAU = 2 * Math.PI;

// ─── Complex arithmetic  (C = [re, im]) ──────────────────────────────────────

type C = [number, number];

function cSub([ar, ai]: C, [br, bi]: C): C {
  return [ar - br, ai - bi];
}
function cMul([ar, ai]: C, [br, bi]: C): C {
  return [ar * br - ai * bi, ar * bi + ai * br];
}
function cDiv([ar, ai]: C, [br, bi]: C): C {
  const d = br * br + bi * bi;
  return [(ar * br + ai * bi) / d, (ai * br - ar * bi) / d];
}
function cConj([re, im]: C): C {
  return [re, -im];
}
function cAbs([re, im]: C): number {
  return Math.sqrt(re * re + im * im);
}
function cArg([re, im]: C): number {
  return Math.atan2(im, re);
}
function cExp([re, im]: C): C {
  const r = Math.exp(re);
  return [r * Math.cos(im), r * Math.sin(im)];
}
function cSqrt(z: C): C {
  const r = Math.sqrt(cAbs(z));
  const phi = cArg(z) / 2;
  return [r * Math.cos(phi), r * Math.sin(phi)];
}

/** Möbius automorphism of the Poincaré disk that swaps 0 and a. */
function pdiskAut(a: C, z: C): C {
  return cDiv(cSub(a, z), cSub([1, 0], cMul(z, cConj(a))));
}

// ─── Grid ────────────────────────────────────────────────────────────────────
//
// A grid is two families of interior curves plus an explicit boundary curve.
//   grid.x[i][j]  — j-th anchor of the i-th vertical   interior curve
//   grid.y[i][j]  — j-th anchor of the i-th horizontal interior curve
//   grid.boundary — anchor points of the closed boundary curve (open polygon,
//                   caller closes it when drawing)

type Pt = [number, number];

interface BezierCurve {
  pts: Pt[];
  // Catmull-Rom control points per segment.
  // For open curves: cp.length === pts.length - 1
  // For closed curves: cp.length === pts.length (last segment wraps to pts[0])
  cp: [Pt, Pt][];
}

interface Grid {
  x: BezierCurve[];
  y: BezierCurve[];
  boundary: BezierCurve;
}

function linspace(lo: number, hi: number, n: number): number[] {
  if (n === 1) return [(lo + hi) / 2];
  return Array.from({ length: n }, (_, i) => lo + (hi - lo) * (i / (n - 1)));
}

// ─── Bezier curve helpers ─────────────────────────────────────────────────────

/** Catmull-Rom control points for an open polyline. */
function computeOpenHandles(pts: Pt[]): [Pt, Pt][] {
  const n = pts.length;
  const cp: [Pt, Pt][] = [];
  for (let i = 0; i < n - 1; i++) {
    const p0 = pts[i > 0 ? i - 1 : i]!;
    const p1 = pts[i]!;
    const p2 = pts[i + 1]!;
    const p3 = pts[i + 2 < n ? i + 2 : i + 1]!;
    cp.push([
      [p1[0] + (p2[0] - p0[0]) / 6, p1[1] + (p2[1] - p0[1]) / 6],
      [p2[0] - (p3[0] - p1[0]) / 6, p2[1] - (p3[1] - p1[1]) / 6],
    ]);
  }
  return cp;
}

/** Catmull-Rom control points for a closed curve (wraps around). */
function computeClosedHandles(pts: Pt[]): [Pt, Pt][] {
  const n = pts.length;
  const cp: [Pt, Pt][] = [];
  for (let i = 0; i < n; i++) {
    const p0 = pts[(i - 1 + n) % n]!;
    const p1 = pts[i]!;
    const p2 = pts[(i + 1) % n]!;
    const p3 = pts[(i + 2) % n]!;
    cp.push([
      [p1[0] + (p2[0] - p0[0]) / 6, p1[1] + (p2[1] - p0[1]) / 6],
      [p2[0] - (p3[0] - p1[0]) / 6, p2[1] - (p3[1] - p1[1]) / 6],
    ]);
  }
  return cp;
}

function makeBezierCurve(pts: Pt[], closed: boolean): BezierCurve {
  return { pts, cp: closed ? computeClosedHandles(pts) : computeOpenHandles(pts) };
}

// ─── Fourier boundary curve ───────────────────────────────────────────────────

// Curve from riemann_mapping.py line 793:
//   x(t) = 0.5·cos t + 0.15·sin 2t
//   y(t) = 0.5·sin t − 0.1·cos 3t
// Scaled so the maximum radius equals TARGET_MAX_R (< 1 so it sits inside
// the unit disk and the Riemann iteration has room to expand it).
const TARGET_MAX_R = 0.86;

function makeFourierBoundary(n: number): Pt[] {
  const raw: Pt[] = Array.from({ length: n }, (_, i) => {
    const t = (TAU * i) / n;
    return [
      0.5 * Math.cos(t) + 0.15 * Math.sin(2 * t),
      0.5 * Math.sin(t) - 0.1 * Math.cos(3 * t),
    ];
  });
  const maxR = Math.max(...raw.map(([x, y]) => Math.sqrt(x * x + y * y)));
  const k = TARGET_MAX_R / maxR;
  return raw.map(([x, y]) => [x * k, y * k]);
}

// ─── Grid construction inside an arbitrary closed polygon ────────────────────

/** y-values where the vertical line x = xv crosses the polygon boundary. */
function findXCrossings(poly: Pt[], xv: number): number[] {
  const crossings: number[] = [];
  const n = poly.length;
  for (let i = 0; i < n; i++) {
    const a = poly[i]!;
    const b = poly[(i + 1) % n]!;
    if ((a[0] <= xv && xv < b[0]) || (b[0] <= xv && xv < a[0])) {
      crossings.push(a[1] + ((xv - a[0]) * (b[1] - a[1])) / (b[0] - a[0]));
    }
  }
  return crossings;
}

/** x-values where the horizontal line y = yv crosses the polygon boundary. */
function findYCrossings(poly: Pt[], yv: number): number[] {
  const crossings: number[] = [];
  const n = poly.length;
  for (let i = 0; i < n; i++) {
    const a = poly[i]!;
    const b = poly[(i + 1) % n]!;
    if ((a[1] <= yv && yv < b[1]) || (b[1] <= yv && yv < a[1])) {
      crossings.push(a[0] + ((yv - a[1]) * (b[0] - a[0])) / (b[1] - a[1]));
    }
  }
  return crossings;
}

/**
 * Builds Cartesian gridlines clipped to the interior of the closed polygon
 * `boundary`.  Crossings are paired (entering / exiting), so this handles
 * non-convex curves without producing segments that cross outside the domain.
 */
function makeGridInsideCurve(
  boundary: Pt[],
  numX: number,
  numY: number,
  numAnchors: number,
): Grid {
  const xs = boundary.map((p) => p[0]);
  const ys = boundary.map((p) => p[1]);
  const xmin = Math.min(...xs),
    xmax = Math.max(...xs);
  const ymin = Math.min(...ys),
    ymax = Math.max(...ys);

  const xGrid: BezierCurve[] = [];
  for (const xv of linspace(xmin, xmax, numX + 2).slice(1, -1)) {
    const yCrossings = findXCrossings(boundary, xv).sort((a, b) => a - b);
    for (let i = 0; i + 1 < yCrossings.length; i += 2) {
      xGrid.push(
        makeBezierCurve(
          linspace(yCrossings[i]!, yCrossings[i + 1]!, numAnchors).map(
            (yv) => [xv, yv] as Pt,
          ),
          false,
        ),
      );
    }
  }

  const yGrid: BezierCurve[] = [];
  for (const yv of linspace(ymin, ymax, numY + 2).slice(1, -1)) {
    const xCrossings = findYCrossings(boundary, yv).sort((a, b) => a - b);
    for (let i = 0; i + 1 < xCrossings.length; i += 2) {
      yGrid.push(
        makeBezierCurve(
          linspace(xCrossings[i]!, xCrossings[i + 1]!, numAnchors).map(
            (xv) => [xv, yv] as Pt,
          ),
          false,
        ),
      );
    }
  }

  return { x: xGrid, y: yGrid, boundary: makeBezierCurve(boundary, true) };
}

function transformGrid(grid: Grid, fn: (p: Pt) => Pt): Grid {
  return {
    x: grid.x.map((c) => makeBezierCurve(c.pts.map(fn), false)),
    y: grid.y.map((c) => makeBezierCurve(c.pts.map(fn), false)),
    boundary: makeBezierCurve(grid.boundary.pts.map(fn), true),
  };
}

function lerpGrid(g1: Grid, g2: Grid, t: number): Grid {
  const lerpPt = (a: Pt, b: Pt): Pt => [
    (1 - t) * a[0] + t * b[0],
    (1 - t) * a[1] + t * b[1],
  ];
  return {
    x: g1.x.map((c, i) =>
      makeBezierCurve(c.pts.map((p, j) => lerpPt(p, g2.x[i]!.pts[j]!)), false),
    ),
    y: g1.y.map((c, i) =>
      makeBezierCurve(c.pts.map((p, j) => lerpPt(p, g2.y[i]!.pts[j]!)), false),
    ),
    boundary: makeBezierCurve(
      g1.boundary.pts.map((p, i) => lerpPt(p, g2.boundary.pts[i]!)),
      true,
    ),
  };
}

function getBoundaryPoints(grid: Grid): Pt[] {
  return grid.boundary.pts;
}

// ─── Branch of √ continuous on the domain ────────────────────────────────────

/**
 * Returns [θ_min, θ_max] such that every boundary point has argument in
 * [θ_min, θ_max].  The interval excludes the largest angular gap so the
 * branch cut of √ falls outside the domain.
 */
function getPhaseBounds(grid: Grid): [number, number] {
  const phases = getBoundaryPoints(grid)
    .map(([x, y]) => Math.atan2(y, x))
    .sort((a, b) => a - b);
  const n = phases.length;

  let maxGap = -1;
  let maxIdx = 0;
  for (let i = 0; i < n; i++) {
    const gap = (phases[(i + 1) % n]! - phases[i]! + TAU) % TAU;
    if (gap > maxGap) {
      maxGap = gap;
      maxIdx = i;
    }
  }

  let thetaMax = phases[maxIdx]!;
  let thetaMin = phases[(maxIdx + 1) % n]!;
  if (thetaMax - thetaMin < 0) thetaMax += TAU;
  if (thetaMin >= Math.PI) {
    thetaMin -= TAU;
    thetaMax -= TAU;
  }
  return [thetaMin, thetaMax];
}

/** Returns the branch of √z that is continuous on the domain covered by grid. */
function makeSqrtFn(grid: Grid): (z: C) => C {
  const [lo, hi] = getPhaseBounds(grid);
  const phi = (lo + hi) / 2;
  const rotIn = cExp([0, -phi]);
  const rotOut = cExp([0, phi / 2]);
  return (z: C) => cMul(cSqrt(cMul(z, rotIn)), rotOut);
}

// ─── Riemann expansion step ──────────────────────────────────────────────────

// TODO Add an option to do N iterations of this function internally
function riemannStep(grid: Grid): Grid {
  const bdy = getBoundaryPoints(grid).filter(([x, y]) => x * x + y * y < 1.0);
  if (bdy.length === 0) return grid;

  const best = bdy.reduce((m, p) =>
    p[0]! * p[0]! + p[1]! * p[1]! < m[0]! * m[0]! + m[1]! * m[1]! ? p : m,
  );
  const norm = Math.sqrt(best[0]! * best[0]! + best[1]! * best[1]!);

  const factor = 1 / Math.pow(norm, 0.2);
  const a: C = [best[0]! * factor, best[1]! * factor];

  const tGrid = transformGrid(grid, ([x, y]) => {
    const r = pdiskAut(a, [x, y]);
    return [r[0]!, r[1]!];
  });
  const sqrtFn = makeSqrtFn(tGrid);

  const pa = cArg(a);
  const sa = sqrtFn(a);

  let rot = cExp([0, pa / 2]);
  const [lo, hi] = getPhaseBounds(tGrid);
  if (Math.abs((lo + hi) / 2 - pa) > Math.PI) {
    rot = cMul(rot, [-1, 0]);
  }

  return transformGrid(grid, ([x, y]) => {
    const z: C = [x, y];
    const r = cMul(pdiskAut(sa, sqrtFn(pdiskAut(a, z))), rot);
    return [r[0]!, r[1]!];
  });
}

// ─── Rendering ───────────────────────────────────────────────────────────────

const PANEL_W = 460;
const PANEL_H = 460;
const GAP = 24;
const CANVAS_W = 2 * PANEL_W + GAP;
const CANVAS_H = PANEL_H;
const SCALE = 210; // pixels per unit (unit circle radius = SCALE pixels)

const LCX = PANEL_W / 2;
const LCY = PANEL_H / 2;
const RCX = PANEL_W + GAP + PANEL_W / 2;
const RCY = PANEL_H / 2;

function drawBezierCurve(
  ctx: CanvasRenderingContext2D,
  curve: BezierCurve,
  cx: number,
  cy: number,
): void {
  const { pts, cp } = curve;
  if (pts.length < 2) return;
  const tx = (p: Pt) => cx + p[0] * SCALE;
  const ty = (p: Pt) => cy - p[1] * SCALE;
  ctx.beginPath();
  ctx.moveTo(tx(pts[0]!), ty(pts[0]!));
  for (let i = 0; i < cp.length; i++) {
    const [c1, c2] = cp[i]!;
    ctx.bezierCurveTo(tx(c1), ty(c1), tx(c2), ty(c2), tx(pts[i + 1]!), ty(pts[i + 1]!));
  }
  ctx.stroke();
}

function drawClosedBezierCurve(
  ctx: CanvasRenderingContext2D,
  curve: BezierCurve,
  cx: number,
  cy: number,
): void {
  const { pts, cp } = curve;
  if (pts.length < 2) return;
  const tx = (p: Pt) => cx + p[0] * SCALE;
  const ty = (p: Pt) => cy - p[1] * SCALE;
  ctx.beginPath();
  ctx.moveTo(tx(pts[0]!), ty(pts[0]!));
  for (let i = 0; i < cp.length; i++) {
    const [c1, c2] = cp[i]!;
    const next = pts[(i + 1) % pts.length]!;
    ctx.bezierCurveTo(tx(c1), ty(c1), tx(c2), ty(c2), tx(next), ty(next));
  }
  ctx.closePath();
  ctx.stroke();
}

function renderGridPanel(
  ctx: CanvasRenderingContext2D,
  grid: Grid,
  cx: number,
  cy: number,
): void {
  ctx.strokeStyle = "rgba(100,160,255,0.4)";
  ctx.lineWidth = 0.75;
  for (const curve of grid.x) drawBezierCurve(ctx, curve, cx, cy);
  for (const curve of grid.y) drawBezierCurve(ctx, curve, cx, cy);

  ctx.strokeStyle = "rgba(100,160,255,1.0)";
  ctx.lineWidth = 2;
  drawClosedBezierCurve(ctx, grid.boundary, cx, cy);
}

// Render the image
function render(
  ctx: CanvasRenderingContext2D,
  initial: Grid,
  current: Grid,
): void {
  ctx.fillStyle = "#0e1117";
  ctx.fillRect(0, 0, CANVAS_W, CANVAS_H);

  // Left panel
  // Static source domain
  renderGridPanel(ctx, initial, LCX, LCY);

  // Right panel
  // Unit circle
  ctx.beginPath();
  ctx.arc(RCX, RCY, SCALE, 0, TAU);
  ctx.strokeStyle = "rgba(255,255,255,0.85)";
  ctx.lineWidth = 2;
  ctx.stroke();

  // Current iteration of the mapped grid
  renderGridPanel(ctx, current, RCX, RCY);
}

// Render the image
function render_intermediate(
  ctx: CanvasRenderingContext2D,
  initial: Grid,
  final: Grid,
  progress: number,
): void {
  ctx.fillStyle = "#0e1117";
  ctx.fillRect(0, 0, CANVAS_W, CANVAS_H);

  // Partial progress between the two grids
  const cx = RCX * progress + LCX * (1 - progress);
  const cy = RCY * progress + LCY * (1 - progress);
  let intermediate = lerpGrid(initial, final, progress);
  renderGridPanel(ctx, intermediate, cx, cy);

  // Right panel
  // Unit circle
  ctx.beginPath();
  ctx.arc(RCX, RCY, SCALE, 0, TAU);
  ctx.strokeStyle = "rgba(255,255,255,0.85)";
  ctx.lineWidth = 2;
  ctx.stroke();
}

// ─── App ─────────────────────────────────────────────────────────────────────

const NUM_ITERS = 500;
const NUM_X = 20;
const NUM_Y = 20;
const BOUNDARY_PTS = 200;
const NUM_ANCHORS = 50;

function buildUI(container: HTMLElement): {
  canvas: HTMLCanvasElement;
  slider: HTMLInputElement;
  label: HTMLElement;
} {
  container.style.cssText =
    "display:flex;flex-direction:column;align-items:center;gap:16px;" +
    "padding:24px;background:#0e1117;border-radius:8px;font-family:monospace;";

  const canvas = document.createElement("canvas");
  canvas.width = CANVAS_W;
  canvas.height = CANVAS_H;
  canvas.style.cssText = "border-radius:4px;max-width:100%;";

  const label = document.createElement("div");
  label.style.cssText = "color:#8b9dc3;font-size:14px;";
  label.textContent = "Computing…";

  const slider = document.createElement("input");
  slider.type = "range";
  slider.min = "0";
  slider.max = "0";
  slider.value = "0";
  slider.disabled = true;
  slider.style.cssText = `width:min(${CANVAS_W}px,90vw);cursor:pointer;accent-color:#64a0ff;`;

  container.append(canvas, label, slider);
  return { canvas, slider, label };
}

function initRiemannApp(containerId: string): void {
  const container = document.getElementById(containerId);
  if (!container) return;

  const { canvas, slider, label } = buildUI(container);
  const ctx = canvas.getContext("2d")!;
  if (!ctx) return;

  const boundary = makeFourierBoundary(BOUNDARY_PTS);
  const initial = makeGridInsideCurve(boundary, NUM_X, NUM_Y, NUM_ANCHORS);

  // Intermediate states stored. Starts with just the initial shape
  const states: Grid[] = [initial];
  // render(ctx, initial, initial);
  render_intermediate(ctx, initial, initial, 0);

  // Track changing state
  let step = 0;
  let state: Grid = initial;
  let final: Grid;

  // TODO Do this rendering for several possible curves
  // TODO Make it possible to drag points on the curve to modify it, and accordingly recompute the final mapping.
  // TODO Instead of storing all the intermediate steps, just compute the final one
  function computeNext(): void {
    // Loop which
    if (step >= NUM_ITERS) {
      console.log("Computation done");
      label.textContent = "Drag slider to map domain onto disk";
      // label.textContent =
      //   "Iteration: 0  —  drag slider to map domain onto disk";
      slider.max = String(NUM_ITERS);
      slider.disabled = false;
      // TODO Instead of rendering the n-th state (out of N) when the slider is pulled along,
      // render (n/N)-of-the-way deformation between the initial state and final state.
      slider.addEventListener("input", () => {
        const n = parseInt(slider.value, 10);
        // label.textContent = `Iteration: ${n} / ${NUM_ITERS}`;
        // render(ctx, initial, states[n]!);
        render_intermediate(ctx, initial, final, n / (NUM_ITERS - 1));
      });
      return;
    }

    // Loop when we're still computing iterations

    // Store the current version
    // states.push(riemannStep(states[step]!));

    // Store just the final version
    state = riemannStep(state);
    if (step == NUM_ITERS - 1) {
      final = state;
    }
    step++;
    slider.max = String(step);
    // render(ctx, initial, states[step]!);
    label.textContent = `Computing… ${Math.round((100 * step) / NUM_ITERS)}%`;
    setTimeout(computeNext, 0);
  }

  setTimeout(computeNext, 0);
}

initRiemannApp("riemann-app");
