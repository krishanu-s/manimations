/**
 * Quick sanity tests for convolve_pdfs.
 * Run with: npx tsx src/pdf-utils.test.ts
 */

import {
  convolve_pdfs,
  buildPDF,
  estimateMean,
  estimateVariance,
  N_FFT,
  DX,
  XSPACE,
} from "./pdf-utils";

const computeMean = estimateMean;
const computeVariance = estimateVariance;

function assert(cond: boolean, msg: string) {
  if (!cond) throw new Error(`FAIL: ${msg}`);
  console.log(`  PASS: ${msg}`);
}

function approxEq(a: number, b: number, tol: number, label: string) {
  assert(
    Math.abs(a - b) < tol,
    `${label}: got ${a.toFixed(6)}, want ${b.toFixed(6)} (tol ${tol})`,
  );
}

// ── Test 1: Gaussian convolution ──────────────────────────────────────────────
// N(0, σ1²) * N(0, σ2²) = N(0, σ1² + σ2²)
{
  console.log("Test 1: convolve two centred Gaussians");
  const s1 = 0.6,
    s2 = 0.8;
  const p1 = buildPDF({ means: [0], stds: [s1], scales: [1] })[0];
  const p2 = buildPDF({ means: [0], stds: [s2], scales: [1] })[0];
  const conv = convolve_pdfs(p1, p2);

  approxEq(computeMean(conv), 0, 0.01, "mean = 0");
  approxEq(
    computeVariance(conv),
    s1 ** 2 + s2 ** 2,
    0.01,
    `variance = ${(s1 ** 2 + s2 ** 2).toFixed(4)}`,
  );

  // Also verify it integrates to ~1
  let norm = 0;
  for (let k = 0; k < N_FFT; k++) norm += conv[k]!;
  approxEq(norm * DX, 1, 0.005, "integrates to 1");
}

// ── Test 2: mean shifts add ───────────────────────────────────────────────────
// N(μ1, σ²) * N(μ2, σ²) should have mean μ1 + μ2
{
  console.log("Test 2: means are additive");
  const mu1 = -1.0,
    mu2 = 0.5,
    s = 0.5;
  const p1 = buildPDF({ means: [mu1], stds: [s], scales: [1] })[0];
  const p2 = buildPDF({ means: [mu2], stds: [s], scales: [1] })[0];
  const conv = convolve_pdfs(p1, p2);

  approxEq(computeMean(conv), mu1 + mu2, 0.01, `mean = ${mu1 + mu2}`);
  approxEq(
    computeVariance(conv),
    2 * s ** 2,
    0.01,
    `variance = ${(2 * s ** 2).toFixed(4)}`,
  );
}

// ── Test 3: convolving with a Dirac (very narrow Gaussian) is identity ────────
{
  console.log("Test 3: convolution with near-Dirac is identity");
  const s = 0.7;
  const p = buildPDF({ means: [0], stds: [s], scales: [1] })[0];
  const d = buildPDF({ means: [0], stds: [0.05], scales: [1] })[0]; // near-Dirac
  const conv = convolve_pdfs(p, d);

  // variance should barely change
  approxEq(computeVariance(conv), s ** 2, 0.01, `variance ≈ ${s ** 2}`);

  // point-wise should be close to p
  let maxDiff = 0;
  for (let k = 0; k < N_FFT; k++)
    maxDiff = Math.max(maxDiff, Math.abs(conv[k]! - p[k]!));
  assert(
    maxDiff < 0.05,
    `max pointwise diff < 0.05 (got ${maxDiff.toFixed(4)})`,
  );
}

// ── Test 4: commutativity ─────────────────────────────────────────────────────
{
  console.log("Test 4: p1 * p2 == p2 * p1");
  const p1 = buildPDF({
    means: [-1, 0.5],
    stds: [0.4, 0.6],
    scales: [0.6, 0.4],
  })[0];
  const p2 = buildPDF({ means: [0.3], stds: [0.8], scales: [1] })[0];
  const c12 = convolve_pdfs(p1, p2);
  const c21 = convolve_pdfs(p2, p1);

  let maxDiff = 0;
  for (let k = 0; k < N_FFT; k++)
    maxDiff = Math.max(maxDiff, Math.abs(c12[k]! - c21[k]!));
  assert(
    maxDiff < 1e-10,
    `max pointwise diff < 1e-10 (got ${maxDiff.toExponential(2)})`,
  );
}

console.log("\nAll tests passed.");
