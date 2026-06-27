/**
 * pdf-utils.tsx — computational primitives used for manipulating probability
 * distributions. Used in CLTInterpolation, OrnsteinUhlenbeck, etc.
 */

import { Matrix, solve } from "ml-matrix";

// ─── Grid format ───────────────────────────────────────────────────────────

export const N_FFT = 512; // number of sample points; must be a power of 2
export const XMIN = -6.0;
export const XMAX = 6.0;
export const DX = (XMAX - XMIN) / N_FFT;
export const XSPACE = Object.freeze(
  Array.from({ length: N_FFT }, (_, k) => XMIN + k * DX),
) as readonly number[];

// Numerically compute the mean of a PDF sampled in grid format
export function estimateMean(pdf: Float64Array): number {
  let ex1 = 0;
  for (let k = 0; k < N_FFT; k++) {
    ex1 += XSPACE[k]! * pdf[k]!;
  }
  return ex1 * DX;
}

// Numerically compute the variance of a PDF sampled in grid format
export function estimateVariance(pdf: Float64Array): number {
  // Added a step in case the PDF has nonzero mean.
  let ex1 = 0;
  let ex2 = 0;
  for (let k = 0; k < N_FFT; k++) {
    ex2 += XSPACE[k]! ** 2 * pdf[k]!;
    ex1 += XSPACE[k]! * pdf[k]!;
  }
  return ex2 * DX - (ex1 * DX) ** 2;
}

// Estimate the information value log(p)
export function estimateInformation(pdf: Float64Array, x: number): number {
  const raw = (x - XMIN) / DX;
  const k = Math.floor(raw);
  if (k < 0 || k >= N_FFT - 1) return -Infinity;
  const frac = raw - k;
  const px = (1 - frac) * pdf[k]! + frac * pdf[k + 1]!;
  return px > 0 ? Math.log(px) : -Infinity;
}

// Numerically compute the expected information (entropy)
export function estimateEntropy(pdf: Float64Array): number {
  let h = 0;
  for (let k = 0; k < N_FFT; k++) {
    const p = pdf[k]!;
    if (p > 0) h -= p * Math.log(p);
  }
  return h * DX;
}

// Estimate the score function d/dx log(p(x))
export function estimateScore(pdf: Float64Array, x: number): number {
  const k = Math.max(1, Math.min(N_FFT - 2, Math.round((x - XMIN) / DX)));
  const px = pdf[k]!;
  if (px < 1e-15) return 0;
  return (pdf[k + 1]! - pdf[k - 1]!) / (2 * DX * px);
}

// Numerically compute the variance of the score function (Fisher information)
export function estimateFisher(pdf: Float64Array): number {
  let fisher = 0;
  for (let k = 1; k < N_FFT - 1; k++) {
    const p = pdf[k]!;
    if (p < 1e-15) continue;
    const dp = (pdf[k + 1]! - pdf[k - 1]!) / (2 * DX);
    fisher += (dp * dp) / p;
  }
  return fisher * DX;
}

// ── FFT (Cooley-Tukey radix-2, in-place) ─────────────────────────────────────

type CPLX = [number, number];

// Computes the Fast Fourier Transform (or its inverse) for a complex array of sampled values
export function fft(re: Float64Array, im: Float64Array, invert: boolean): void {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      let t = re[i]!;
      re[i] = re[j]!;
      re[j] = t;
      t = im[i]!;
      im[i] = im[j]!;
      im[j] = t;
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (invert ? 1 : -1) * ((2 * Math.PI) / len);
    const wr = Math.cos(ang),
      wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cr = 1,
        ci = 0;
      for (let j = 0; j < len >> 1; j++) {
        const u = i + j,
          v = i + j + (len >> 1);
        const vr = re[v]! * cr - im[v]! * ci;
        const vi = re[v]! * ci + im[v]! * cr;
        re[v] = re[u]! - vr;
        im[v] = im[u]! - vi;
        re[u] = re[u]! + vr;
        im[u] = im[u]! + vi;
        const nc = cr * wr - ci * wi;
        ci = cr * wi + ci * wr;
        cr = nc;
      }
    }
  }
  if (invert)
    for (let i = 0; i < n; i++) {
      re[i]! /= n;
      im[i]! /= n;
    }
}

// Complex power via polar form: (a + ib)^n
export function cplxPow(a: number, b: number, n: number): [number, number] {
  const r = Math.sqrt(a * a + b * b);
  if (r < 1e-30) return [0, 0];
  const th = Math.atan2(b, a) * n;
  return [r ** n * Math.cos(th), r ** n * Math.sin(th)];
}

// Compute the convolution of two grid-sampled PDFs, using FFT.
// Uses the convolution theorem: F[p1 * p2] = F[p1] · F[p2].
// The DX factor converts the discrete circular convolution into a Riemann-sum
// approximation of the continuous convolution integral.
export function convolve_pdfs(
  pdf1: Float64Array,
  pdf2: Float64Array,
): Float64Array {
  const re1 = new Float64Array(pdf1);
  const im1 = new Float64Array(N_FFT);
  fft(re1, im1, false);

  const re2 = new Float64Array(pdf2);
  const im2 = new Float64Array(N_FFT);
  fft(re2, im2, false);

  const cRe = new Float64Array(N_FFT);
  const cIm = new Float64Array(N_FFT);
  for (let j = 0; j < N_FFT; j++) {
    const s = j % 2 === 0 ? 1 : -1;
    cRe[j] = (re1[j]! * re2[j]! - im1[j]! * im2[j]!) * DX * s;
    cIm[j] = (re1[j]! * im2[j]! + im1[j]! * re2[j]!) * DX * s;
  }

  fft(cRe, cIm, true);

  let norm = 0;
  for (let k = 0; k < N_FFT; k++) {
    cRe[k] = Math.max(0, cRe[k]!);
    norm += cRe[k]!;
  }
  norm *= DX;
  if (norm > 1e-15) for (let k = 0; k < N_FFT; k++) cRe[k]! /= norm;

  return cRe;
}

// ── Gaussian PDF construction ──────────────────────────────────────────────────

// PDF which is a linear combination of the PDFs of Gaussians. For such PDFs, many computations
// can be done *exactly* and so it can be advantageous to keep them in this form. For example,
// - The characteristic function E[exp(iωX)] when X = N(m, σ) is exp(imω)exp(-σ²ω²/2), and thus
// has a similar basis format.
// This is the format used for user-defined PDFs formed by dragging control points around.
interface GaussianSumPDF {
  means: readonly number[]; // Mean values of the Gaussians
  stds: readonly number[]; // Std values of the Gaussians
  scales: readonly number[]; // Scalings of the Gaussians
}

// Exact computation of the information log(p(x))
export function computeInformation(pdf: GaussianSumPDF, x: number): number {
  let sumH = 0,
    sumP = 0;
  for (let i = 0; i < pdf.means.length; i++) {
    const h = pdf.scales[i]!;
    sumH += h;
    sumP += h * gaussPDF(x, pdf.means[i]!, pdf.stds[i]!);
  }
  if (sumH < 1e-15 || sumP < 1e-15) return -Infinity;
  return Math.log(sumP / sumH);
}

// Exact computation of the expected information (entropy), via numerical integration
// using the exact p(x) formula. No closed form exists for Gaussian mixtures in general.
export function computeEntropy(pdf: GaussianSumPDF): number {
  let sumH = 0;
  for (let i = 0; i < pdf.scales.length; i++) sumH += pdf.scales[i]!;
  if (sumH < 1e-15) return 0;
  let h = 0;
  for (let k = 0; k < N_FFT; k++) {
    const x = XSPACE[k]!;
    let px = 0;
    for (let i = 0; i < pdf.means.length; i++)
      px += pdf.scales[i]! * gaussPDF(x, pdf.means[i]!, pdf.stds[i]!);
    px /= sumH;
    if (px > 0) h -= px * Math.log(px);
  }
  return h * DX;
}

// Exact computation of the score function p'(x)/p(x).
// Derivative of a Gaussian mixture: d/dx N(x;μ,σ²) = N(x;μ,σ²) · (-(x-μ)/σ²).
// The score is then a weighted average of -(x-μᵢ)/σᵢ² with weights hᵢ·N(x;μᵢ,σᵢ²).
export function computeScore(pdf: GaussianSumPDF, x: number): number {
  let sumP = 0,
    sumDP = 0;
  for (let i = 0; i < pdf.means.length; i++) {
    const h = pdf.scales[i]!,
      mu = pdf.means[i]!,
      sigma = pdf.stds[i]!;
    const g = h * gaussPDF(x, mu, sigma);
    sumP += g;
    sumDP += g * (-(x - mu) / (sigma * sigma));
  }
  if (sumP < 1e-15) return 0;
  return sumDP / sumP;
}

// Exact computation of the Fisher information I = ∫ (p'(x))² / p(x) dx,
// via numerical integration using the exact p(x) and p'(x) formulas.
export function computeFisher(pdf: GaussianSumPDF): number {
  let sumH = 0;
  for (let i = 0; i < pdf.scales.length; i++) sumH += pdf.scales[i]!;
  if (sumH < 1e-15) return 0;
  let fisher = 0;
  for (let k = 0; k < N_FFT; k++) {
    const x = XSPACE[k]!;
    let px = 0,
      dpx = 0;
    for (let i = 0; i < pdf.means.length; i++) {
      const h = pdf.scales[i]!,
        mu = pdf.means[i]!,
        sigma = pdf.stds[i]!;
      const g = h * gaussPDF(x, mu, sigma);
      px += g;
      dpx += g * (-(x - mu) / (sigma * sigma));
    }
    px /= sumH;
    dpx /= sumH;
    if (px > 1e-15) fisher += (dpx * dpx) / px;
  }
  return fisher * DX;
}

// Computes the convolution of two gaussian sum pdfs using explicit formula
export function convolveGaussians(
  g1: GaussianSumPDF,
  g2: GaussianSumPDF,
): GaussianSumPDF {
  // For every pair of Gaussian terms, we have
  // c1N(m1, σ1²) * c2N(m2, σ2²) = c1c2N(m1 + m2, σ1² + σ2²)
  // Get quadratic blowup each time we apply this, so don't do it too many times.
  const n1 = g1.means.length;
  const n2 = g2.means.length;
  let m1: number, s1: number, c1: number;
  let means: number[] = [];
  let stds: number[] = [];
  let scales: number[] = [];
  for (let i1 = 0; i1 < n1; i1++) {
    m1 = g1.means[i1]!;
    s1 = g1.stds[i1]!;
    c1 = g1.scales[i1]!;
    for (let i2 = 0; i2 < n2; i2++) {
      means.push(m1 + g2.means[i2]!);
      stds.push(Math.sqrt(s1 ** 2 + g2.stds[i2]! ** 2));
      scales.push(c1 * g2.scales[i2]!);
    }
  }

  return {
    means: means,
    stds: stds,
    scales: scales,
  };
}

// Computes the characteristic function E[exp(iω·X)] for a gaussian sum pdf X.
//   φ_X(ω) = exp(−σ_b²ω²/2) · [Σᵢ hᵢ exp(iω·xᵢ)] / [Σᵢ hᵢ]
// Evaluating this directly at any ω avoids the interpolation error that appears
// when reading a DFT-sampled charFn at non-integer frequency indices.
export function evalCharFn(
  gaussian_sum_pdf: GaussianSumPDF,
  omega: number,
): CPLX {
  let sumH = 0,
    sumCos = 0,
    sumSin = 0;
  let l = gaussian_sum_pdf.means.length;
  let std: number, mean: number, h: number;
  for (let i = 0; i < l; i++) {
    mean = gaussian_sum_pdf.means[i]!;
    std = gaussian_sum_pdf.stds[i]!;
    h = gaussian_sum_pdf.scales[i]!;
    sumH += h;
    sumCos +=
      h * Math.cos(omega * mean) * Math.exp(-0.5 * std * std * omega * omega);
    sumSin +=
      h * Math.sin(omega * mean) * Math.exp(-0.5 * std * std * omega * omega);
  }
  if (sumH < 1e-15) return [1, 0];
  return [sumCos / sumH, sumSin / sumH];
}

// Gaussian PDF with a given mean and std
export function gaussPDF(x: number, mu: number, sigma: number): number {
  const z = (x - mu) / sigma;
  return Math.exp(-0.5 * z * z) / (sigma * Math.sqrt(2 * Math.PI));
}

// Given a set of desired pdf values p(x1), p(x2), ... p(xn) at control points
// x1, x2, ..., xn, produces a sequence of scales c1, c2, ..., cn such that
// sum_j cj * N(xj, σj^2)(xi) = p(xi) for all i
export function interpPDFfromValues(
  means: readonly number[],
  stds: readonly number[],
  p_values: readonly number[],
): GaussianSumPDF {
  const n = means.length;
  // First generate the n x n matrix A_ij = N(xj, σj^2)(xi)
  const A = new Matrix(
    Array.from({ length: n }, (_, i) =>
      Array.from({ length: n }, (_, j) =>
        gaussPDF(means[i]!, means[j]!, stds[j]!),
      ),
    ),
  );
  // Then use c = A^{-1]p
  return {
    means: means,
    stds: stds,
    scales: solve(A, Matrix.columnVector(p_values)).to1DArray(),
  };
}

// Builds the PDF in grid format as a linear combination of Gaussians scaled.
// Also includes the normalization factor int (c1G1(x) + ... + cnGn(x) dx)
export function buildPDF(gaussian_pdf: GaussianSumPDF): [Float64Array, number] {
  const pdf = new Float64Array(N_FFT);
  let norm = 0;
  for (let k = 0; k < N_FFT; k++) {
    const x = XSPACE[k]!;
    let v = 0;
    let n_ctrl = gaussian_pdf.means.length;
    for (let i = 0; i < n_ctrl; i++)
      v +=
        gaussian_pdf.scales[i]! *
        gaussPDF(x, gaussian_pdf.means[i]!, gaussian_pdf.stds[i]!);
    pdf[k] = v;
    norm += v;
  }
  norm *= DX;
  if (norm > 1e-15) for (let k = 0; k < N_FFT; k++) pdf[k]! /= norm;
  // Output norm so that we can scale
  return [pdf, norm];
}
