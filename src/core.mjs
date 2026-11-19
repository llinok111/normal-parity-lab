// core.mjs — point-cloud normal orientation with an orientation-parity certificate.
//
// Pure ES module: no three.js, no DOM, no I/O, no environment. Every function is a
// deterministic function of its arguments.
//
// Pipeline: procedural cloud -> grid kNN -> PCA normal -> MST sign propagation ->
// non-tree-edge parity audit (is the field globally consistent?) -> score vs analytic truth.
//
// All tolerances are RELATIVE to the cloud bounding-box diagonal. There is no absolute
// epsilon that depends on a chosen unit, so the same cloud scaled by 1e-6 or 1e6 produces
// the same graph decisions (verified in bench/verify.mjs).

export const K_DEFAULT = 12;
export const SHAPES = ['sphere', 'torus', 'cylinder', 'box', 'saddle', 'mobius'];

// ---------------------------------------------------------------- deterministic RNG
export function mulberry32(seed) {
  let a = seed >>> 0;
  return function next() {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// 12-uniform sum: sigma ~= 1, bounded, and bit-reproducible.
function gauss(rand) {
  let s = 0;
  for (let i = 0; i < 12; i++) s += rand();
  return s - 6;
}

// ---------------------------------------------------------------- procedural corpus
// Each shape returns a point + its analytic unit normal. `pos` is uniform in area (or a
// disclosed proxy for the saddle), so kNN density is comparable across shapes.
function sampleShape(shape, rand, p) {
  const TAU = Math.PI * 2;
  if (shape === 'sphere') {
    const r = p.r;
    const u = rand() * TAU;
    const w = rand() * 2 - 1;          // uniform in cos(theta)
    const s = Math.sqrt(Math.max(0, 1 - w * w));
    return [[r * s * Math.cos(u), r * s * Math.sin(u), r * w], [s * Math.cos(u), s * Math.sin(u), w]];
  }
  if (shape === 'torus') {
    const { R, r } = p;
    const u = rand() * TAU;
    let v = 0;
    for (let guard = 0; guard < 64; guard++) {
      const c = rand() * TAU;
      if (rand() * (R + r) <= R + r * Math.cos(c)) { v = c; break; }
    }
    const cu = Math.cos(u), su = Math.sin(u), cv = Math.cos(v), sv = Math.sin(v);
    const rad = R + r * cv;
    return [[rad * cu, rad * su, r * sv], [cv * cu, cv * su, sv]];
  }
  if (shape === 'cylinder') {
    const { r, h } = p;
    const u = rand() * TAU;
    const z = (rand() - 0.5) * h;
    return [[r * Math.cos(u), r * Math.sin(u), z], [Math.cos(u), Math.sin(u), 0]];
  }
  if (shape === 'box') {
    const h = p.h;
    const faces = [
      [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
    ];
    const f = faces[Math.min(5, Math.floor(rand() * 6))];
    const m = h * 0.92; // inset so the analytic normal is unambiguous at the sample itself
    const a = (rand() * 2 - 1) * m;
    const b = (rand() * 2 - 1) * m;
    if (f[0] !== 0) return [[f[0] * h, a, b], f];
    if (f[1] !== 0) return [[a, f[1] * h, b], f];
    return [[a, b, f[2] * h], f];
  }
  if (shape === 'saddle') {
    const { a: A } = p;
    const x = rand() * 2 - 1;
    const y = rand() * 2 - 1;
    const z = A * (x * x - y * y);
    const nx = -2 * A * x, ny = 2 * A * y, nz = 1;
    const L = Math.hypot(nx, ny, nz);
    return [[x, y, z], [nx / L, ny / L, nz / L]];
  }
  if (shape === 'mobius') {
    const { R, w } = p;
    const u = rand() * TAU;
    const v = (rand() * 2 - 1) * w;
    const c = Math.cos(u / 2), s = Math.sin(u / 2);
    const rad = R + v * c;
    const P = [rad * Math.cos(u), rad * Math.sin(u), v * s];
    // central-difference tangents; the band is smooth and the step is relative to R
    const e = R * 1e-6;
    const at = (uu, vv) => {
      const cc = Math.cos(uu / 2), ss = Math.sin(uu / 2), rr = R + vv * cc;
      return [rr * Math.cos(uu), rr * Math.sin(uu), vv * ss];
    };
    const a1 = at(u + e, v), a0 = at(u - e, v);
    const b1 = at(u, v + e), b0 = at(u, v - e);
    const du = [(a1[0] - a0[0]) / (2 * e), (a1[1] - a0[1]) / (2 * e), (a1[2] - a0[2]) / (2 * e)];
    const dv = [(b1[0] - b0[0]) / (2 * e), (b1[1] - b0[1]) / (2 * e), (b1[2] - b0[2]) / (2 * e)];
    const nx = du[1] * dv[2] - du[2] * dv[1];
    const ny = du[2] * dv[0] - du[0] * dv[2];
    const nz = du[0] * dv[1] - du[1] * dv[0];
    const L = Math.hypot(nx, ny, nz) || 1;
    return [P, [nx / L, ny / L, nz / L]];
  }
  throw new Error('unknown shape: ' + shape);
}

export const DEFAULTS = {
  sphere: { r: 1.0 },
  torus: { R: 1.0, r: 0.36 },
  cylinder: { r: 0.72, h: 1.7 },
  box: { h: 0.75 },
  saddle: { a: 0.55 },
  mobius: { R: 1.0, w: 0.32 },
};

/** Build one deterministic cloud. Returns flat Float64Arrays. */
export function buildCloud(spec) {
  const shape = spec.shape;
  if (!SHAPES.includes(shape)) throw new Error('unknown shape: ' + shape);
  const n = spec.n | 0;
  const params = Object.assign({}, DEFAULTS[shape], spec.params || {});
  const rand = mulberry32(spec.seed >>> 0);
  const pts = new Float64Array(n * 3);
  const truth = new Float64Array(n * 3);
  let lo = [Infinity, Infinity, Infinity];
  let hi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < n; i++) {
    const [p, nrm] = sampleShape(shape, rand, params);
    pts[3 * i] = p[0]; pts[3 * i + 1] = p[1]; pts[3 * i + 2] = p[2];
    truth[3 * i] = nrm[0]; truth[3 * i + 1] = nrm[1]; truth[3 * i + 2] = nrm[2];
    for (let k = 0; k < 3; k++) { if (p[k] < lo[k]) lo[k] = p[k]; if (p[k] > hi[k]) hi[k] = p[k]; }
  }
  const diag = Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]);
  const spacing = n > 1 ? diag / Math.sqrt(n) : 0;
  const noise = spec.noise || 0;
  if (noise > 0) {
    const sigma = noise * spacing;
    for (let i = 0; i < n; i++) {
      pts[3 * i] += gauss(rand) * sigma;
      pts[3 * i + 1] += gauss(rand) * sigma;
      pts[3 * i + 2] += gauss(rand) * sigma;
    }
  }
  return { shape, n, points: pts, truth, diag, spacing, noise };
}

// ---------------------------------------------------------------- grid kNN
function bboxOf(points, n) {
  let lo = [Infinity, Infinity, Infinity];
  let hi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < 3; k++) {
      const v = points[3 * i + k];
      if (v < lo[k]) lo[k] = v;
      if (v > hi[k]) hi[k] = v;
    }
  }
  return { lo, hi, diag: Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) };
}

function hashCell(ix, iy, iz) {
  return (Math.imul(ix, 73856093) ^ Math.imul(iy, 19349663) ^ Math.imul(iz, 83492791)) | 0;
}

export function knnGraph(points, n, k = K_DEFAULT) {
  const bb = bboxOf(points, n);
  const diag = bb.diag;
  let cell = diag * Math.sqrt(k / n);
  if (!(cell > 0) || !isFinite(cell)) cell = 1;
  const grid = new Map();
  const ci = new Int32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const ix = Math.floor((points[3 * i] - bb.lo[0]) / cell);
    const iy = Math.floor((points[3 * i + 1] - bb.lo[1]) / cell);
    const iz = Math.floor((points[3 * i + 2] - bb.lo[2]) / cell);
    ci[3 * i] = ix; ci[3 * i + 1] = iy; ci[3 * i + 2] = iz;
    const key = hashCell(ix, iy, iz);
    let bucket = grid.get(key);
    if (!bucket) { bucket = []; grid.set(key, bucket); }
    bucket.push(i);
  }
  const neighbors = new Int32Array(n * k).fill(-1);
  const dists = new Float64Array(n * k).fill(Infinity);
  const bestD = new Float64Array(k);
  const bestI = new Int32Array(k);
  let maxCand = 0, maxRing = 0;
  for (let i = 0; i < n; i++) {
    bestD.fill(Infinity); bestI.fill(-1);
    const xi = ci[3 * i], yi = ci[3 * i + 1], zi = ci[3 * i + 2];
    const px = points[3 * i], py = points[3 * i + 1], pz = points[3 * i + 2];
    let count = 0, cand = 0;
    for (let ring = 0; ring <= 12; ring++) {
      for (let dx = -ring; dx <= ring; dx++) {
        for (let dy = -ring; dy <= ring; dy++) {
          for (let dz = -ring; dz <= ring; dz++) {
            if (ring > 0 && Math.max(Math.abs(dx), Math.abs(dy), Math.abs(dz)) !== ring) continue;
            const bucket = grid.get(hashCell(xi + dx, yi + dy, zi + dz));
            if (!bucket) continue;
            for (let t = 0; t < bucket.length; t++) {
              const j = bucket[t];
              if (j === i) continue;
              // buckets are keyed by a 32-bit hash: reject colliding cells exactly
              if (ci[3 * j] !== xi + dx || ci[3 * j + 1] !== yi + dy || ci[3 * j + 2] !== zi + dz) continue;
              const ex = points[3 * j] - px, ey = points[3 * j + 1] - py, ez = points[3 * j + 2] - pz;
              const d = ex * ex + ey * ey + ez * ez;
              cand++;
              if (count === k && d >= bestD[k - 1]) continue;
              let pos = count < k ? count : k - 1;
              while (pos > 0 && bestD[pos - 1] > d) {
                bestD[pos] = bestD[pos - 1]; bestI[pos] = bestI[pos - 1]; pos--;
              }
              bestD[pos] = d; bestI[pos] = j;
              if (count < k) count++;
            }
          }
        }
      }
      maxRing = Math.max(maxRing, ring);
      // any cell outside rings 0..ring is at least ring*cell away: safe stop
      if (count === k && (ring * cell) * (ring * cell) >= bestD[k - 1]) break;
    }
    if (cand > maxCand) maxCand = cand;
    for (let t = 0; t < count; t++) { neighbors[i * k + t] = bestI[t]; dists[i * k + t] = bestD[t]; }
  }
  return { neighbors, dists, k, cell, maxRing, maxCand };
}

/** Independent oracle: exact O(n^2) kNN, no grid, no epsilon. */
export function bruteKnn(points, n, k = K_DEFAULT, queries = null) {
  const qs = queries || Array.from({ length: n }, (_, i) => i);
  const out = [];
  for (const qi of qs) {
    const px = points[3 * qi], py = points[3 * qi + 1], pz = points[3 * qi + 2];
    const d = [];
    for (let j = 0; j < n; j++) {
      if (j === qi) continue;
      const ex = points[3 * j] - px, ey = points[3 * j + 1] - py, ez = points[3 * j + 2] - pz;
      d.push([ex * ex + ey * ey + ez * ez, j]);
    }
    d.sort((a, b) => (a[0] - b[0]) || (a[1] - b[1]));
    out.push(d.slice(0, k));
  }
  return out;
}

// ---------------------------------------------------------------- PCA normals
function jacobiEigen3(m) {
  const a = [[m[0], m[1], m[2]], [m[1], m[3], m[4]], [m[2], m[4], m[5]]];
  const v = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  let scale = Math.abs(a[0][0]) + Math.abs(a[1][1]) + Math.abs(a[2][2]);
  if (!(scale > 0)) scale = 1;
  for (let sweep = 0; sweep < 24; sweep++) {
    const off = Math.abs(a[0][1]) + Math.abs(a[0][2]) + Math.abs(a[1][2]);
    if (off <= 1e-15 * scale) break;
    for (const [p, q] of [[0, 1], [0, 2], [1, 2]]) {
      const apq = a[p][q];
      if (Math.abs(apq) <= 1e-18 * scale) continue;
      const tau = (a[q][q] - a[p][p]) / (2 * apq);
      const t = (tau >= 0 ? 1 : -1) / (Math.abs(tau) + Math.sqrt(tau * tau + 1));
      const c = 1 / Math.sqrt(t * t + 1);
      const s = t * c;
      for (let k = 0; k < 3; k++) {
        if (k === p || k === q) continue;
        const akp = a[k][p], akq = a[k][q];
        a[k][p] = c * akp - s * akq;
        a[p][k] = a[k][p];
        a[k][q] = s * akp + c * akq;
        a[q][k] = a[k][q];
      }
      const app = a[p][p], aqq = a[q][q];
      a[p][p] = app - t * apq;
      a[q][q] = aqq + t * apq;
      a[p][q] = 0; a[q][p] = 0;
      for (let k = 0; k < 3; k++) {
        const vkp = v[k][p], vkq = v[k][q];
        v[k][p] = c * vkp - s * vkq;
        v[k][q] = s * vkp + c * vkq;
      }
    }
  }
  const vals = [a[0][0], a[1][1], a[2][2]];
  let mn = 0;
  for (let i = 1; i < 3; i++) if (vals[i] < vals[mn]) mn = i;
  return { values: vals, normal: [v[0][mn], v[1][mn], v[2][mn]] };
}

export function pcaNormals(points, n, neighbors, k = K_DEFAULT) {
  const out = new Float64Array(n * 3);
  const nbr = Math.max(4, k);
  for (let i = 0; i < n; i++) {
    let cx = 0, cy = 0, cz = 0, c = 0;
    for (let t = 0; t < k; t++) {
      const j = neighbors[i * k + t];
      if (j < 0) break;
      cx += points[3 * j]; cy += points[3 * j + 1]; cz += points[3 * j + 2]; c++;
    }
    if (c === 0) { out[3 * i + 2] = 1; continue; }
    cx /= c; cy /= c; cz /= c;
    let m00 = 0, m01 = 0, m02 = 0, m11 = 0, m12 = 0, m22 = 0;
    for (let t = 0; t < k; t++) {
      const j = neighbors[i * k + t];
      if (j < 0) break;
      const dx = points[3 * j] - cx, dy = points[3 * j + 1] - cy, dz = points[3 * j + 2] - cz;
      m00 += dx * dx; m01 += dx * dy; m02 += dx * dz;
      m11 += dy * dy; m12 += dy * dz; m22 += dz * dz;
    }
    const e = jacobiEigen3([m00, m01, m02, m11, m12, m22]);
    let [x, y, z] = e.normal;
    const L = Math.hypot(x, y, z) || 1;
    out[3 * i] = x / L; out[3 * i + 1] = y / L; out[3 * i + 2] = z / L;
    void nbr;
  }
  return out;
}

// ---------------------------------------------------------------- orientation
// Prim over the undirected kNN graph, edge weight = 1 - |dot(n_i, n_j)| (Hoppe-style).
// Each vertex takes the sign of its parent, so every tree edge ends with dot > 0.
export function orientByMST(points, n, neighbors, normals, k = K_DEFAULT) {
  const visited = new Uint8Array(n);
  const parent = new Int32Array(n).fill(-1);
  const best = new Float64Array(n).fill(Infinity);
  const out = Float64Array.from(normals);
  const tree = new Set();
  const heap = [];
  const push = (w, v) => {
    heap.push([w, v]);
    let i = heap.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (heap[p][0] <= heap[i][0]) break;
      const t = heap[p]; heap[p] = heap[i]; heap[i] = t; i = p;
    }
  };
  const pop = () => {
    const top = heap[0];
    const last = heap.pop();
    if (heap.length) {
      heap[0] = last;
      let i = 0;
      for (;;) {
        const l = 2 * i + 1, r = l + 1;
        let m = i;
        if (l < heap.length && heap[l][0] < heap[m][0]) m = l;
        if (r < heap.length && heap[r][0] < heap[m][0]) m = r;
        if (m === i) break;
        const t = heap[m]; heap[m] = heap[i]; heap[i] = t; i = m;
      }
    }
    return top;
  };
  let components = 0;
  let flips = 0;
  let seeds = 0;
  // centroid, used only to pick the global sign of each component's seed
  let gx = 0, gy = 0, gz = 0;
  for (let i = 0; i < n; i++) { gx += points[3 * i]; gy += points[3 * i + 1]; gz += points[3 * i + 2]; }
  gx /= n; gy /= n; gz /= n;
  let component = new Int32Array(n).fill(-1);
  for (let s = 0; s < n; s++) {
    if (visited[s]) continue;
    components++;
    // seed = the unvisited point farthest from the centroid (a stable, shape-agnostic pick)
    let seed = -1, bestD = -1;
    for (let i = 0; i < n; i++) {
      if (visited[i]) continue;
      const dx = points[3 * i] - gx, dy = points[3 * i + 1] - gy, dz = points[3 * i + 2] - gz;
      const d = dx * dx + dy * dy + dz * dz;
      if (d > bestD) { bestD = d; seed = i; }
    }
    // orient seed outward from the centroid; this only fixes the component's global sign
    const sx = points[3 * seed] - gx, sy = points[3 * seed + 1] - gy, sz = points[3 * seed + 2] - gz;
    const d0 = out[3 * seed] * sx + out[3 * seed + 1] * sy + out[3 * seed + 2] * sz;
    if (d0 < 0) {
      out[3 * seed] = -out[3 * seed]; out[3 * seed + 1] = -out[3 * seed + 1]; out[3 * seed + 2] = -out[3 * seed + 2];
      seeds++;
    }
    best[seed] = 0;
    push(0, seed);
    while (heap.length) {
      const [, u] = pop();
      if (visited[u]) continue;
      visited[u] = 1;
      component[u] = components - 1;
      if (parent[u] >= 0) {
        const p = parent[u];
        const d = out[3 * u] * out[3 * p] + out[3 * u + 1] * out[3 * p + 1] + out[3 * u + 2] * out[3 * p + 2];
        if (d < 0) {
          out[3 * u] = -out[3 * u]; out[3 * u + 1] = -out[3 * u + 1]; out[3 * u + 2] = -out[3 * u + 2];
          flips++;
        }
        tree.add(u < p ? u * n + p : p * n + u);
      }
      for (let t = 0; t < k; t++) {
        const v = neighbors[u * k + t];
        if (v < 0 || visited[v]) continue;
        const dot = out[3 * u] * normals[3 * v] + out[3 * u + 1] * normals[3 * v + 1] + out[3 * u + 2] * normals[3 * v + 2];
        const w = 1 - Math.abs(dot);
        if (w < best[v]) { best[v] = w; parent[v] = u; push(w, v); }
      }
    }
  }
  void seeds;
  return { normals: out, tree, parent, components, flips };
}

/** Parity audit: every non-tree graph edge must join two normals with dot > 0.
 *  A non-zero count means the field is frustrated (impossible to orient consistently). */
export function parityAudit(n, neighbors, normals, tree, k = K_DEFAULT) {
  const seen = new Set();
  let edges = 0, nonTree = 0, frustrated = 0;
  const frustratedList = [];
  for (let i = 0; i < n; i++) {
    for (let t = 0; t < k; t++) {
      const j = neighbors[i * k + t];
      if (j < 0) continue;
      const a = i < j ? i : j, b = i < j ? j : i;
      const key = a * n + b;
      if (seen.has(key)) continue;
      seen.add(key);
      edges++;
      if (tree.has(key)) continue;
      nonTree++;
      const d = normals[3 * a] * normals[3 * b] + normals[3 * a + 1] * normals[3 * b + 1] + normals[3 * a + 2] * normals[3 * b + 2];
      if (d < 0) { frustrated++; if (frustratedList.length < 64) frustratedList.push([a, b, d]); }
    }
  }
  return { edges, nonTree, frustrated, rate: nonTree ? frustrated / nonTree : 0, frustratedList };
}

/** Baseline: each point's sign is decided independently by "point away from the centroid". */
export function centroidOutward(points, n, normals) {
  let gx = 0, gy = 0, gz = 0;
  for (let i = 0; i < n; i++) { gx += points[3 * i]; gy += points[3 * i + 1]; gz += points[3 * i + 2]; }
  gx /= n; gy /= n; gz /= n;
  const out = Float64Array.from(normals);
  for (let i = 0; i < n; i++) {
    const dx = points[3 * i] - gx, dy = points[3 * i + 1] - gy, dz = points[3 * i + 2] - gz;
    if (out[3 * i] * dx + out[3 * i + 1] * dy + out[3 * i + 2] * dz < 0) {
      out[3 * i] = -out[3 * i]; out[3 * i + 1] = -out[3 * i + 1]; out[3 * i + 2] = -out[3 * i + 2];
    }
  }
  return out;
}

// ---------------------------------------------------------------- scoring
const RAD2DEG = 180 / Math.PI;

/** Angle error + sign error after the single best global flip (majority vote).
 *  The global flip is the only truth-dependent step and is applied to every method alike. */
export function score(normals, truth, n) {
  const angles = new Float64Array(n);
  let agree = 0;
  for (let i = 0; i < n; i++) {
    const d = normals[3 * i] * truth[3 * i] + normals[3 * i + 1] * truth[3 * i + 1] + normals[3 * i + 2] * truth[3 * i + 2];
    if (d >= 0) agree++;
    angles[i] = Math.acos(Math.min(1, Math.max(-1, Math.abs(d)))) * RAD2DEG;
  }
  const signErrorRate = Math.min(agree, n - agree) / n;
  const sorted = Array.from(angles).sort((a, b) => a - b);
  const q = (p) => sorted[Math.min(sorted.length - 1, Math.floor(p * (sorted.length - 1)))];
  let sum = 0;
  for (const a of sorted) sum += a;
  return {
    angles, medianAngleDeg: q(0.5), meanAngleDeg: sum / sorted.length,
    p95AngleDeg: q(0.95), maxAngleDeg: sorted[sorted.length - 1], signErrorRate,
  };
}

/** End-to-end pipeline for one spec. Timings are filled by bench, not here. */
export function pipeline(spec, opts = {}) {
  const k = opts.k || K_DEFAULT;
  const cloud = buildCloud(spec);
  const { points, truth, n } = cloud;
  const t0 = nowMs();
  const knn = knnGraph(points, n, k);
  const t1 = nowMs();
  const pca = pcaNormals(points, n, knn.neighbors, k);
  const t2 = nowMs();
  const oriented = orientByMST(points, n, knn.neighbors, pca, k);
  const t3 = nowMs();
  const parity = parityAudit(n, knn.neighbors, oriented.normals, oriented.tree, k);
  const t4 = nowMs();
  const sc = score(oriented.normals, truth, n);
  const baseline = score(centroidOutward(points, n, pca), truth, n);
  return {
    id: spec.id, shape: spec.shape, seed: spec.seed, n, k, noise: cloud.noise,
    components: oriented.components, flips: oriented.flips,
    medianAngleDeg: sc.medianAngleDeg, meanAngleDeg: sc.meanAngleDeg,
    p95AngleDeg: sc.p95AngleDeg, maxAngleDeg: sc.maxAngleDeg,
    signErrorRate: sc.signErrorRate,
    baselineSignErrorRate: baseline.signErrorRate,
    baselineMedianAngleDeg: baseline.medianAngleDeg,
    nonTreeEdges: parity.nonTree, frustratedEdges: parity.frustrated, frustrationRate: parity.rate,
    edges: parity.edges,
    knnMs: t1 - t0, pcaMs: t2 - t1, orientMs: t3 - t2, parityMs: t4 - t3, coreMs: t4 - t0,
    angles: sc.angles, baselineAngles: baseline.angles,
  };
}

export function nowMs() {
  const p = typeof globalThis !== 'undefined' ? globalThis.performance : undefined;
  if (p && typeof p.now === 'function') return p.now();
  return Date.now();
}

// ---------------------------------------------------------------- corpus helpers
export function makeCorpus() {
  const specs = [];
  const variants = 8;
  for (let s = 0; s < SHAPES.length; s++) {
    for (let v = 0; v < variants; v++) {
      const shape = SHAPES[s];
      const spec = {
        id: shape + '-' + String(v).padStart(2, '0'),
        shape,
        seed: 100003 * (s + 1) + 7919 * v + 13,
        n: 3000,
        noise: 0,
        params: {},
      };
      if (v >= 6) spec.noise = 0.02; // two of eight variants carry 2% spacing noise
      specs.push(spec);
    }
  }
  return specs;
}

/** Stable 53-bit fingerprint of a cloud (quantized to 1e-9 of the bbox diagonal). */
export function cloudHash(cloud) {
  const { points, truth, n, diag } = cloud;
  const q = 1e-9 * (diag || 1);
  let h1 = 0x811c9dc5, h2 = 0x01000193;
  const mix = (x) => {
    h1 = Math.imul(h1 ^ (x & 0xffffffff), 16777619) >>> 0;
    h2 = Math.imul(h2 ^ ((x >>> 16) | 0), 2246822519) >>> 0;
  };
  for (let i = 0; i < n * 3; i++) {
    mix(Math.round(points[i] / q) | 0);
    mix(Math.round(truth[i] * 1e9) | 0);
  }
  return (h1 >>> 0).toString(16).padStart(8, '0') + (h2 >>> 0).toString(16).padStart(8, '0');
}

/** Pooled statistics over many pipeline results. */
export function pool(results) {
  const all = [];
  let signErr = 0, baseErr = 0, pts = 0, nonTree = 0, frus = 0, clouds = 0;
  let worst = null;
  for (const r of results) {
    clouds++;
    for (const a of r.angles) all.push(a);
    signErr += r.signErrorRate * r.n;
    baseErr += r.baselineSignErrorRate * r.n;
    pts += r.n;
    nonTree += r.nonTreeEdges;
    frus += r.frustratedEdges;
    if (!worst || r.p95AngleDeg > worst.p95AngleDeg) worst = r;
  }
  all.sort((a, b) => a - b);
  const q = (p) => all[Math.min(all.length - 1, Math.floor(p * (all.length - 1)))];
  let sum = 0;
  for (const a of all) sum += a;
  return {
    clouds, points: pts,
    medianAngleDeg: q(0.5), meanAngleDeg: sum / all.length, p95AngleDeg: q(0.95), maxAngleDeg: all[all.length - 1],
    signErrorPoints: Math.round(signErr), signErrorRate: signErr / pts,
    baselineSignErrorPoints: Math.round(baseErr), baselineSignErrorRate: baseErr / pts,
    nonTreeEdges: nonTree, frustratedEdges: frus, frustrationRate: nonTree ? frus / nonTree : 0,
    worstCloud: worst ? worst.id : null,
  };
}