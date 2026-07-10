// Utilities for gas simulations

// ── Colors ────────────────────────────────────────────────────────────────────

export const COL = {
  // for visual
  particle: "rgba(30, 100, 220, 0.78)",
  particleEdge: "rgba(15, 60, 160, 0.90)",
  wall: "#555", // wall
  vdiv: "rgba(40, 40, 40, 0.2)", // virtual divider
  bg: "#f8f9fb", // background

  // for histograms
  barFill: "rgba(40, 110, 230, 0.35)", // fill of a bar in a histogram
  barEdge: "rgba(30, 90, 210, 0.82)", // boundary of a bar in a histogram
  theory: "rgba(210, 40, 60, 0.90)", // theoretical curve overlaid on histogram
  axis: "#555",
  text: "#333",
  dim: "#666",
} as const;

// ── Initialization ────────────────────────────────────────────────────────────

// Produces a vector of the given length whose angle is chosen uniformly in [0, 2π]
export function randomRadialVector(r: number): [number, number] {
  const θ = Math.random() * 2 * Math.PI;
  return [r * Math.cos(θ), r * Math.sin(θ)];
}

// Produces a vector of the given length whose angle is chosen uniformly in [π/4, 3π/4, 5π/4, 7π/4]
export function randomDiagonalVector(r: number): [number, number] {
  const θ = ((Math.floor(4 * Math.random()) + 0.5) * Math.PI) / 2;
  return [r * Math.cos(θ), r * Math.sin(θ)];
}
