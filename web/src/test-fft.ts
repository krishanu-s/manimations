/**
 * test-fft.ts — verifies the FFT and characteristic function pipeline
 *
 * Run with:  npx tsx src/test-fft.ts
 *        or: node --input-type=module < <(npx esbuild src/test-fft.ts --bundle --platform=node)
 */

// ─── FFT (verbatim copy from CLTInterpolation.tsx) ────────────────────────────

function fft(re: Float64Array, im: Float64Array, invert: boolean): void {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      let t = re[i]!; re[i] = re[j]!; re[j] = t;
      t = im[i]!; im[i] = im[j]!; im[j] = t;
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (invert ? 1 : -1) * (2 * Math.PI / len);
    const wr = Math.cos(ang), wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let j = 0; j < (len >> 1); j++) {
        const u = i + j, v = i + j + (len >> 1);
        const vr = re[v]! * cr - im[v]! * ci;
        const vi = re[v]! * ci + im[v]! * cr;
        re[v] = re[u]! - vr; im[v] = im[u]! - vi;
        re[u] = re[u]! + vr; im[u] = im[u]! + vi;
        const nc = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = nc;
      }
    }
  }
  if (invert) for (let i = 0; i < n; i++) { re[i]! /= n; im[i]! /= n; }
}

// ─── Pipeline helpers (verbatim copy) ─────────────────────────────────────────

const N_FFT = 512;
const XMIN = -6.0;
const XMAX = 6.0;
const DX = (XMAX - XMIN) / N_FFT;
const XSPACE = Array.from({ length: N_FFT }, (_, k) => XMIN + k * DX);

function gaussPDF(x: number, mu: number, sigma: number): number {
  const z = (x - mu) / sigma;
  return Math.exp(-0.5 * z * z) / (sigma * Math.sqrt(2 * Math.PI));
}

function computeCharFn(pdf: Float64Array): [Float64Array, Float64Array] {
  const re = Float64Array.from(pdf);
  const im = new Float64Array(N_FFT);
  fft(re, im, false);
  const cRe = new Float64Array(N_FFT);
  const cIm = new Float64Array(N_FFT);
  for (let j = 0; j < N_FFT; j++) {
    const s = j % 2 === 0 ? 1 : -1;
    cRe[j] = DX * s * re[j]!;
    cIm[j] = DX * s * (-im[j]!);
  }
  return [cRe, cIm];
}

function interpChar(cRe: Float64Array, cIm: Float64Array, si: number): [number, number] {
  const idx = ((si % N_FFT) + N_FFT) % N_FFT;
  const i0 = Math.floor(idx);
  const i1 = (i0 + 1) % N_FFT;
  const f = idx - i0;
  return [
    cRe[i0]! + f * (cRe[i1]! - cRe[i0]!),
    cIm[i0]! + f * (cIm[i1]! - cIm[i0]!),
  ];
}

function cplxPow(a: number, b: number, n: number): [number, number] {
  const r = Math.sqrt(a * a + b * b);
  if (r < 1e-30) return [0, 0];
  const th = Math.atan2(b, a) * n;
  return [r ** n * Math.cos(th), r ** n * Math.sin(th)];
}

function computeOutputPDF(
  cRe: Float64Array, cIm: Float64Array,
  sigma2: number, n: number, t: number,
): Float64Array {
  const alpha = Math.sqrt(Math.max(0, 1 - t)) / Math.sqrt(n);
  const aRe = new Float64Array(N_FFT);
  const aIm = new Float64Array(N_FFT);
  for (let j = 0; j < N_FFT; j++) {
    const sj = j <= (N_FFT >> 1) ? j : j - N_FFT;
    const [xr, xi] = interpChar(cRe, cIm, sj * alpha);
    const [pr, pi] = cplxPow(xr, xi, n);
    const omega = 2 * Math.PI * sj / (N_FFT * DX);
    const gf = Math.exp(-0.5 * t * sigma2 * omega * omega);
    const s = j % 2 === 0 ? 1 : -1;
    aRe[j] = s * pr * gf;
    aIm[j] = s * pi * gf;
  }
  fft(aRe, aIm, false);
  const out = new Float64Array(N_FFT);
  for (let k = 0; k < N_FFT; k++) out[k] = Math.max(0, aRe[k]! / (N_FFT * DX));
  return out;
}

function computeVariance(pdf: Float64Array): number {
  let v = 0;
  for (let k = 0; k < N_FFT; k++) v += XSPACE[k]! ** 2 * pdf[k]!;
  return v * DX;
}

// ─── Test helpers ──────────────────────────────────────────────────────────────

function pass(label: string) { console.log(`  PASS  ${label}`); }
function fail(label: string, detail: string) { console.log(`  FAIL  ${label}: ${detail}`); }

function check(label: string, got: number, expected: number, tol = 1e-9) {
  if (Math.abs(got - expected) <= tol) pass(label);
  else fail(label, `got ${got.toExponential(4)}, expected ${expected.toExponential(4)}, err = ${Math.abs(got - expected).toExponential(3)}`);
}

// ─── TEST 1: FFT of Fourier basis elements ────────────────────────────────────
//
// The m-th Fourier basis element is e_m[k] = e^{2πimk/N}.
// Its DFT: FFT[e_m][j] = Σ_k e^{2πimk/N} · e^{-2πijk/N} = N · δ[j=m].
// Test for several values of m.

console.log("\nTEST 1: FFT of Fourier basis elements e^{2πimk/N}");
{
  const N = 16;
  for (const m of [0, 1, 3, 7, 8, 14]) {
    const re = new Float64Array(N);
    const im = new Float64Array(N);
    for (let k = 0; k < N; k++) {
      re[k] = Math.cos(2 * Math.PI * m * k / N);
      im[k] = Math.sin(2 * Math.PI * m * k / N);
    }
    fft(re, im, false);
    // Expect: re[m] = N, im[m] = 0, everything else = 0
    let maxErrRe = 0, maxErrIm = 0;
    for (let j = 0; j < N; j++) {
      maxErrRe = Math.max(maxErrRe, Math.abs(re[j]! - (j === m ? N : 0)));
      maxErrIm = Math.max(maxErrIm, Math.abs(im[j]!));
    }
    check(`  m=${m}: spike at j=${m}, re error`, maxErrRe, 0, 1e-9);
    check(`  m=${m}: imaginary parts all zero`, maxErrIm, 0, 1e-9);
  }
}

// ─── TEST 2: IFFT ∘ FFT = identity ────────────────────────────────────────────

console.log("\nTEST 2: IFFT(FFT(f)) = f");
{
  const N = 32;
  const origRe = Float64Array.from({ length: N }, (_, k) => Math.sin(k) + 0.5 * Math.cos(2.7 * k));
  const re = Float64Array.from(origRe);
  const im = new Float64Array(N);
  fft(re, im, false);
  fft(re, im, true);
  let maxErr = 0;
  for (let k = 0; k < N; k++) maxErr = Math.max(maxErr, Math.abs(re[k]! - origRe[k]!), Math.abs(im[k]!));
  check("max round-trip error", maxErr, 0, 1e-12);
}

// ─── TEST 3: characteristic function of a Gaussian ────────────────────────────
//
// If p_X = N(0, σ²) then φ_X(ω) = exp(-σ² ω² / 2).
// We test this at several frequencies via computeCharFn.

console.log("\nTEST 3: characteristic function of N(0,1) vs exp(-ω²/2)");
{
  const SIGMA = 1.0;
  const pdf = new Float64Array(N_FFT);
  for (let k = 0; k < N_FFT; k++) pdf[k] = gaussPDF(XSPACE[k]!, 0, SIGMA);
  // Normalise numerically
  let norm = 0; for (let k = 0; k < N_FFT; k++) norm += pdf[k]!; norm *= DX;
  for (let k = 0; k < N_FFT; k++) pdf[k]! /= norm;

  const [cRe, cIm] = computeCharFn(pdf);

  // Sample at several signed frequency indices (small j where φ is non-negligible)
  for (const j of [0, 1, 2, 4, 8]) {
    const omega = 2 * Math.PI * j / (N_FFT * DX);
    const expected = Math.exp(-0.5 * SIGMA * SIGMA * omega * omega);
    const got = cRe[j]!;  // for symmetric real pdf, imaginary part ≈ 0
    const gotIm = cIm[j]!;
    check(`  j=${j}: Re φ_X(ω_${j}) ≈ exp(-ω²/2)`, got, expected, 5e-4);
    check(`  j=${j}: Im φ_X(ω_${j}) ≈ 0`, gotIm, 0, 1e-10);
  }
}

// ─── TEST 4: pipeline round-trip (n=1, t=0 → output = input) ─────────────────
//
// For n=1, t=0: Y = X₁, so p_Y should equal p_X.

console.log("\nTEST 4: pipeline round-trip n=1, t=0 → output = input");
{
  const SIGMA = 1.0;
  const pdf = new Float64Array(N_FFT);
  for (let k = 0; k < N_FFT; k++) pdf[k] = gaussPDF(XSPACE[k]!, 0, SIGMA);
  let norm = 0; for (let k = 0; k < N_FFT; k++) norm += pdf[k]!; norm *= DX;
  for (let k = 0; k < N_FFT; k++) pdf[k]! /= norm;

  const sigma2 = computeVariance(pdf);
  const [cRe, cIm] = computeCharFn(pdf);
  const out = computeOutputPDF(cRe, cIm, sigma2, 1, 0);

  let maxErr = 0;
  for (let k = 0; k < N_FFT; k++) maxErr = Math.max(maxErr, Math.abs(out[k]! - pdf[k]!));
  check("max pointwise error vs input pdf", maxErr, 0, 1e-4);
  const varOut = computeVariance(out);
  check(`variance preserved (input σ²=${sigma2.toFixed(4)}, output σ²=${varOut.toFixed(4)})`, varOut, sigma2, 5e-4);
}

// ─── TEST 5: CLT (n=4, t=0 → variance should equal σ² of X) ─────────────────
//
// For n=4, t=0: Y = (X₁+…+X₄)/2. If X ~ N(0,1), then Y ~ N(0,1) as well,
// so Var(Y) should be σ² = 1.

console.log("\nTEST 5: CLT invariance (Gaussian input, n=4, t=0)");
{
  const SIGMA = 1.0;
  const pdf = new Float64Array(N_FFT);
  for (let k = 0; k < N_FFT; k++) pdf[k] = gaussPDF(XSPACE[k]!, 0, SIGMA);
  let norm = 0; for (let k = 0; k < N_FFT; k++) norm += pdf[k]!; norm *= DX;
  for (let k = 0; k < N_FFT; k++) pdf[k]! /= norm;

  const sigma2 = computeVariance(pdf);   // ≈ 1
  const [cRe, cIm] = computeCharFn(pdf);
  const out = computeOutputPDF(cRe, cIm, sigma2, 4, 0);

  const varOut = computeVariance(out);
  const normOut = out.reduce((s, v) => s + v, 0) * DX;
  console.log(`    input  σ² = ${sigma2.toFixed(6)}`);
  console.log(`    output σ² = ${varOut.toFixed(6)}  (expected ≈ ${sigma2.toFixed(6)})`);
  console.log(`    output normalisation = ${normOut.toFixed(6)}  (expected ≈ 1)`);
  check("variance unchanged by CLT", varOut, sigma2, 1e-3);
}

// ─── TEST 6: t=1 → pure Gaussian regardless of n ─────────────────────────────
//
// For t=1: Y = G ~ N(0, σ²), so the output should be N(0, σ²) independent of n.

console.log("\nTEST 6: t=1 → pure Gaussian (should be exact for Gaussian input)");
{
  const SIGMA = 1.0;
  const pdf = new Float64Array(N_FFT);
  for (let k = 0; k < N_FFT; k++) pdf[k] = gaussPDF(XSPACE[k]!, 0, SIGMA);
  let norm = 0; for (let k = 0; k < N_FFT; k++) norm += pdf[k]!; norm *= DX;
  for (let k = 0; k < N_FFT; k++) pdf[k]! /= norm;

  const sigma2 = computeVariance(pdf);
  const [cRe, cIm] = computeCharFn(pdf);
  const out = computeOutputPDF(cRe, cIm, sigma2, 4, 1);

  let maxErr = 0;
  for (let k = 0; k < N_FFT; k++) maxErr = Math.max(maxErr, Math.abs(out[k]! - pdf[k]!));
  const varOut = computeVariance(out);
  console.log(`    output σ² = ${varOut.toFixed(6)}  (expected ≈ ${sigma2.toFixed(6)})`);
  check("max pointwise error vs N(0,1)", maxErr, 0, 1e-3);
}

// ─── TEST 7: analytical φ_X for Gaussian mixture — CLT invariance ─────────────
//
// The fix replaces FFT-based interpolation with the exact analytical formula:
//   φ_X(ω) = exp(−σ_b²ω²/2) · [Σᵢ hᵢ exp(iωxᵢ)] / [Σᵢ hᵢ]
// Valid because buildPDF always produces this mixture.
// CLT: for n=4, t=0, the output PDF of Y=(X₁+…+X₄)/2 should have the same
// variance as X.

const CTRL_XS_T = [-3, -2, -1, 0, 1, 2, 3];
const BASIS_STD_T = 0.6;
const N_CTRL_T = 7;

function evalCharFn(heights: readonly number[], omega: number): [number, number] {
  const gf = Math.exp(-0.5 * BASIS_STD_T * BASIS_STD_T * omega * omega);
  let sumH = 0, sumCos = 0, sumSin = 0;
  for (let i = 0; i < N_CTRL_T; i++) {
    const h = heights[i]!;
    sumH += h;
    sumCos += h * Math.cos(omega * CTRL_XS_T[i]!);
    sumSin += h * Math.sin(omega * CTRL_XS_T[i]!);
  }
  if (sumH < 1e-15) return [1, 0];
  return [gf * sumCos / sumH, gf * sumSin / sumH];
}

function computeOutputPDFFixed(
  heights: readonly number[],
  sigma2: number, n: number, t: number,
): Float64Array {
  const alpha = Math.sqrt(Math.max(0, 1 - t)) / Math.sqrt(n);
  const aRe = new Float64Array(N_FFT);
  const aIm = new Float64Array(N_FFT);
  for (let j = 0; j < N_FFT; j++) {
    const sj = j <= (N_FFT >> 1) ? j : j - N_FFT;
    const omega = 2 * Math.PI * sj / (N_FFT * DX);
    const [xr, xi] = evalCharFn(heights, alpha * omega);
    const [pr, pi] = cplxPow(xr, xi, n);
    const gf = Math.exp(-0.5 * t * sigma2 * omega * omega);
    const s = j % 2 === 0 ? 1 : -1;
    aRe[j] = s * pr * gf;
    aIm[j] = s * pi * gf;
  }
  fft(aRe, aIm, false);
  const out = new Float64Array(N_FFT);
  for (let k = 0; k < N_FFT; k++) out[k] = Math.max(0, aRe[k]! / (N_FFT * DX));
  return out;
}

function buildPDF_test(heights: readonly number[]): Float64Array {
  const pdf = new Float64Array(N_FFT);
  let norm = 0;
  for (let k = 0; k < N_FFT; k++) {
    const x = XSPACE[k]!;
    let v = 0;
    for (let i = 0; i < N_CTRL_T; i++) v += heights[i]! * gaussPDF(x, CTRL_XS_T[i]!, BASIS_STD_T);
    pdf[k] = v;
    norm += v;
  }
  norm *= DX;
  if (norm > 1e-15) for (let k = 0; k < N_FFT; k++) pdf[k]! /= norm;
  return pdf;
}

console.log("\nTEST 7: analytical charFn — CLT invariance with Gaussian mixture input");
{
  // Default bimodal heights from the visualization
  const heights = [0.05, 0.4, 1.2, 0.05, 1.2, 0.4, 0.05] as const;
  const pdf = buildPDF_test(heights);
  const sigma2 = computeVariance(pdf);

  for (const n of [1, 2, 4, 10]) {
    const out = computeOutputPDFFixed(heights, sigma2, n, 0);
    const varOut = computeVariance(out);
    const normOut = out.reduce((s, v) => s + v, 0) * DX;
    const ok = Math.abs(varOut - sigma2) < 1e-3;
    console.log(`  n=${n}: input σ²=${sigma2.toFixed(6)}, output σ²=${varOut.toFixed(6)}, norm=${normOut.toFixed(6)}  ${ok ? "PASS" : "FAIL"}`);
  }

  // Also check t=1: output should be pure Gaussian N(0, sigma2)
  const outT1 = computeOutputPDFFixed(heights, sigma2, 4, 1.0);
  const varT1 = computeVariance(outT1);
  check("t=1 → pure Gaussian, variance preserved", varT1, sigma2, 1e-3);
}

// ─── DIAGNOSIS: interpolation error at j=1..4, n=4, t=0 ──────────────────────
//
// For n=4, alpha=0.5: we look up charFn at half-integer indices.
// charFn[j] = φ_X(ω_j) where ω_j = 2πj/(N·DX).
// The true value at ω_j/2 is φ_X(ω_j/2) = exp(-ω_j²/8).
// Linear interpolation between charFn[0] and charFn[1] at fraction 0.5
// gives (charFn[0]+charFn[1])/2, which underestimates the concave-down curve.

console.log("\nDIAGNOSIS: interpolation error at small j, n=4 (alpha=0.5)");
{
  const SIGMA = 1.0;
  const pdf = new Float64Array(N_FFT);
  for (let k = 0; k < N_FFT; k++) pdf[k] = gaussPDF(XSPACE[k]!, 0, SIGMA);
  let norm = 0; for (let k = 0; k < N_FFT; k++) norm += pdf[k]!; norm *= DX;
  for (let k = 0; k < N_FFT; k++) pdf[k]! /= norm;

  const [cRe, cIm] = computeCharFn(pdf);
  const alpha = 0.5;  // n=4, t=0

  for (const j of [1, 2, 4, 8]) {
    const omega = 2 * Math.PI * j / (N_FFT * DX);
    const [interp] = interpChar(cRe, cIm, j * alpha);
    const truePhiAtScaled = Math.exp(-0.5 * SIGMA * SIGMA * (omega * alpha) ** 2);
    const storedAtJ = cRe[j]!;  // φ_X(ω_j) — what the OUTPUT φ_Y should equal
    console.log(`  j=${j}: interp φ_X(ω/2)=${interp.toFixed(6)}, ` +
      `true=${truePhiAtScaled.toFixed(6)}, err=${(interp-truePhiAtScaled).toFixed(6)}` +
      `  →  interp^4=${(interp**4).toFixed(6)}, expected φ_Y(ω_${j})=${storedAtJ.toFixed(6)}`);
  }
}

console.log("");
