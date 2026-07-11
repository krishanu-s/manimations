// Physics simulation utils

// State of a single particle
export type Particle = { x: number; y: number; vx: number; vy: number };

// Global settings for simulation
export interface SimSettings {
  ps: Particle[]; // Gas particles
  r: number; // Particle radius
  g: number; // Gravitational constant
  boxW: number; // Box width
  boxH: number; // Box height
}

// ── Physics ───────────────────────────────────────────────────────────
// Physics for particles in a rectangular box.
export function physicsStepBox({ ps, r, g, boxW, boxH }: SimSettings): void {
  const n = ps.length;
  const dmin = 2 * r;
  const dmin2 = dmin * dmin;

  // Move position and enact gravity, in a way that preserves PE + KE.
  for (let i = 0; i < n; i++) {
    ps[i]!.vy += g / 2;
    ps[i]!.x += ps[i]!.vx;
    ps[i]!.y += ps[i]!.vy;
    ps[i]!.vy += g / 2;
  }

  // Particle-box collisions
  for (let i = 0; i < n; i++) {
    const p = ps[i]!;
    if (p.x < r) {
      p.x = 2 * r - p.x;
      p.vx = Math.abs(p.vx);
    }
    if (p.x > boxW - r) {
      p.x = 2 * (boxW - r) - p.x;
      p.vx = -Math.abs(p.vx);
    }
    if (p.y < r) {
      p.y = 2 * r - p.y;
      p.vy = Math.abs(p.vy);
    }
    if (p.y > boxH - r) {
      p.y = 2 * (boxH - r) - p.y;
      p.vy = -Math.abs(p.vy);
    }
  }

  // Particle-particle collisions
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const a = ps[i]!,
        b = ps[j]!;
      const dx = b.x - a.x,
        dy = b.y - a.y;
      const d2 = dx * dx + dy * dy;
      if (d2 >= dmin2 || d2 < 1e-12) continue;
      const d = Math.sqrt(d2);
      const nx = dx / d,
        ny = dy / d;
      const dvn = (a.vx - b.vx) * nx + (a.vy - b.vy) * ny;
      if (dvn <= 0) continue;
      a.vx -= dvn * nx;
      a.vy -= dvn * ny;
      b.vx += dvn * nx;
      b.vy += dvn * ny;
      const half = 0.5 * (dmin - d);
      a.x -= half * nx;
      a.y -= half * ny;
      b.x += half * nx;
      b.y += half * ny;
    }
  }
}

// Physics for a box with chambers arranged horizontally.
export function physicsStepChambers(
  { ps, r, g, boxW, boxH }: SimSettings,
  walls: readonly number[],
  gapY0: number,
  gapY1: number,
): void {
  const n = ps.length;
  const dmin = 2 * r;
  const dmin2 = dmin * dmin;

  for (let i = 0; i < n; i++) {
    ps[i]!.vx -= g / 2;
    ps[i]!.x += ps[i]!.vx;
    ps[i]!.y += ps[i]!.vy;
    ps[i]!.vx -= g / 2;
  }

  for (let i = 0; i < n; i++) {
    const p = ps[i]!;
    if (p.x < r) {
      p.x = 2 * r - p.x;
      p.vx = Math.abs(p.vx);
    }
    if (p.x > boxW - r) {
      p.x = 2 * (boxW - r) - p.x;
      p.vx = -Math.abs(p.vx);
    }
    if (p.y < r) {
      p.y = 2 * r - p.y;
      p.vy = Math.abs(p.vy);
    }
    if (p.y > boxH - r) {
      p.y = 2 * (boxH - r) - p.y;
      p.vy = -Math.abs(p.vy);
    }

    // Interior chamber walls: impermeable except through the aperture gap.
    for (const wx of walls) {
      if (p.x <= wx - r || p.x >= wx + r) continue;
      const clearsGap = p.y - r >= gapY0 && p.y + r <= gapY1;
      if (clearsGap) continue;
      if (p.vx >= 0) {
        p.x = 2 * (wx - r) - p.x;
        p.vx = -Math.abs(p.vx);
      } else {
        p.x = 2 * (wx + r) - p.x;
        p.vx = Math.abs(p.vx);
      }
    }
  }

  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const a = ps[i]!,
        b = ps[j]!;
      const dx = b.x - a.x,
        dy = b.y - a.y;
      const d2 = dx * dx + dy * dy;
      if (d2 >= dmin2 || d2 < 1e-12) continue;
      const d = Math.sqrt(d2);
      const nx = dx / d,
        ny = dy / d;
      const dvn = (a.vx - b.vx) * nx + (a.vy - b.vy) * ny;
      if (dvn <= 0) continue;
      a.vx -= dvn * nx;
      a.vy -= dvn * ny;
      b.vx += dvn * nx;
      b.vy += dvn * ny;
      const half = 0.5 * (dmin - d);
      a.x -= half * nx;
      a.y -= half * ny;
      b.x += half * nx;
      b.y += half * ny;
    }
  }
}

// Physics for a box with chambers arranged vertically
export function physicsStepChambersVertical(
  { ps, r, g, boxW, boxH }: SimSettings,
  walls: readonly number[],
  gapX0: number,
  gapX1: number,
): void {
  const n = ps.length;
  const dmin = 2 * r;
  const dmin2 = dmin * dmin;

  // Move position and enact gravity, in a way that preserves PE + KE.
  for (let i = 0; i < n; i++) {
    ps[i]!.vy += g / 2;
    ps[i]!.x += ps[i]!.vx;
    ps[i]!.y += ps[i]!.vy;
    ps[i]!.vy += g / 2;
  }

  // Particle-box collisions
  for (let i = 0; i < n; i++) {
    const p = ps[i]!;
    if (p.x < r) {
      p.x = 2 * r - p.x;
      p.vx = Math.abs(p.vx);
    }
    if (p.x > boxW - r) {
      p.x = 2 * (boxW - r) - p.x;
      p.vx = -Math.abs(p.vx);
    }
    if (p.y < r) {
      p.y = 2 * r - p.y;
      p.vy = Math.abs(p.vy);
    }
    if (p.y > boxH - r) {
      p.y = 2 * (boxH - r) - p.y;
      p.vy = -Math.abs(p.vy);
    }

    // Interior chamber walls: impermeable except through the aperture gap.
    for (const wy of walls) {
      if (p.y <= wy - r || p.y >= wy + r) continue;
      const clearsGap = p.x - r >= gapX0 && p.x + r <= gapX1;
      if (clearsGap) continue;
      if (p.vy >= 0) {
        p.y = 2 * (wy - r) - p.y;
        p.vy = -Math.abs(p.vy);
      } else {
        p.y = 2 * (wy + r) - p.y;
        p.vy = Math.abs(p.vy);
      }
    }
  }

  // Particle-particle collisions
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const a = ps[i]!,
        b = ps[j]!;
      const dx = b.x - a.x,
        dy = b.y - a.y;
      const d2 = dx * dx + dy * dy;
      if (d2 >= dmin2 || d2 < 1e-12) continue;
      const d = Math.sqrt(d2);
      const nx = dx / d,
        ny = dy / d;
      const dvn = (a.vx - b.vx) * nx + (a.vy - b.vy) * ny;
      if (dvn <= 0) continue;
      a.vx -= dvn * nx;
      a.vy -= dvn * ny;
      b.vx += dvn * nx;
      b.vy += dvn * ny;
      const half = 0.5 * (dmin - d);
      a.x -= half * nx;
      a.y -= half * ny;
      b.x += half * nx;
      b.y += half * ny;
    }
  }
}
