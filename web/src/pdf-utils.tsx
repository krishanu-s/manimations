/**
 * pdf-utils.tsx — computational primitives used for manipulating probability
 * distributions. Used in CLTInterpolation, OrnsteinUhlenbeck, etc.
 */

// ─── Grid format ───────────────────────────────────────────────────────────

export const N_FFT = 512; // number of sample points; must be a power of 2
export const XMIN = -6.0;
export const XMAX = 6.0;
export const DX = (XMAX - XMIN) / N_FFT;
export const XSPACE = Object.freeze(
  Array.from({ length: N_FFT }, (_, k) => XMIN + k * DX),
) as readonly number[];

// Numerically compute the mean of a PDF sampled in grid format
export function computeMean(pdf: Float64Array): number {
  let ex1 = 0;
  for (let k = 0; k < N_FFT; k++) {
    ex1 += XSPACE[k]! * pdf[k]!;
  }
  return ex1 * DX;
}

// Numerically compute the variance of a PDF sampled in grid format
export function computeVariance(pdf: Float64Array): number {
  // Added a step in case the PDF has nonzero mean.
  let ex1 = 0;
  let ex2 = 0;
  for (let k = 0; k < N_FFT; k++) {
    ex2 += XSPACE[k]! ** 2 * pdf[k]!;
    ex1 += XSPACE[k]! * pdf[k]!;
  }
  return ex2 * DX - (ex1 * DX) ** 2;
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

// ── PDF construction ───────────────────────────────────────────────────────────

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

// Computes the convolution of two gaussian sum pdfs using explicit formula
export function convolveGaussians(
  g1: GaussianSumPDF,
  g2: GaussianSumPDF,
): GaussianSumPDF {
  // For every pair of Gaussian terms, we have
  // c1N(m1, σ1²) * c2N(m2, σ2²) = c1c2N(m1 + m2, σ1² + σ2²)
  // Get quadratic blowup each time we apply this, so don't do it too many times.
  // TODO
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

// Builds the PDF in grid format as a linear combination of Gaussians scaled
export function buildPDF(gaussian_pdf: GaussianSumPDF): Float64Array {
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
  return pdf;
}
