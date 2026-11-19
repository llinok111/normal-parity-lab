// bench/verify.mjs — executable assertions. Exit 0 only if every gate passes.
// Every printed number comes from this run; nothing is read from a cached result file.
import assert from 'node:assert/strict';
import {
  makeCorpus, buildCloud, cloudHash, pipeline, pool, knnGraph, bruteKnn,
  parityAudit, orientByMST, pcaNormals, score, K_DEFAULT, SHAPES,
} from '../src/core.mjs';

const t0 = Date.now();
const say = (...a) => console.log(...a);
const specs = makeCorpus();
const results = specs.map((s) => pipeline(s));
const orientable = results.filter((r) => r.shape !== 'mobius');
const mobius = results.filter((r) => r.shape === 'mobius');
const p = pool(orientable);
const mob = pool(mobius);

say('== corpus ==');
say(`clouds=${specs.length} points=${specs.reduce((a, s) => a + s.n, 0)} k=${K_DEFAULT}`);
assert.equal(specs.length, 48);
assert.equal(specs.reduce((a, s) => a + s.n, 0), 144000);

say('\n== G-A orientation consistency (5 orientable families, 40 clouds) ==');
say(`non-tree edges=${p.nonTreeEdges} frustrated=${p.frustratedEdges} rate=${(p.frustrationRate * 100).toFixed(4)}%`);
say(`components: ${[...new Set(orientable.map((r) => r.components))].join(',')}`);
say(`signError=${p.signErrorPoints}/${p.points} (${(p.signErrorRate * 100).toFixed(4)}%)`);
say(`angle deg: median=${p.medianAngleDeg.toFixed(4)} mean=${p.meanAngleDeg.toFixed(4)} p95=${p.p95AngleDeg.toFixed(3)} max=${p.maxAngleDeg.toFixed(2)}`);
assert.equal(p.frustratedEdges, 0, 'orientable clouds must have zero frustrated edges');
assert.ok(orientable.every((r) => r.components === 1), 'each orientable cloud must be one graph component');
assert.equal(p.signErrorPoints, 0, 'every point must take the correct side up to one global flip');
assert.ok(p.medianAngleDeg < 3, `pooled median angle ${p.medianAngleDeg} must be < 3 deg`);
assert.ok(p.p95AngleDeg < 15, `pooled p95 angle ${p.p95AngleDeg} must be < 15 deg`);
assert.ok(p.meanAngleDeg < 3, `pooled mean angle ${p.meanAngleDeg} must be < 3 deg`);

say('\n== G-B negative control: Mobius band must be frustrated ==');
say(`clouds=${mob.clouds} non-tree=${mob.nonTreeEdges} frustrated=${mob.frustratedEdges} rate=${(mob.frustrationRate * 100).toFixed(3)}%`);
say(`per cloud: ${mobius.map((r) => r.frustratedEdges).join(',')}`);
assert.ok(mobius.every((r) => r.frustratedEdges > 0), 'every Mobius cloud must expose at least one frustrated edge');
assert.ok(mob.frustratedEdges > 40, `expected a clear signal, got ${mob.frustratedEdges}`);
// the auditor is not "always positive": the orientable side is exactly zero
assert.equal(p.frustratedEdges, 0);

say('\n== G-C baseline: per-point centroid-outward sign ==');
say(`centroid sign errors=${p.baselineSignErrorPoints}/${p.points} (${(p.baselineSignErrorRate * 100).toFixed(2)}%)`);
const byShape = {};
for (const r of orientable) {
  (byShape[r.shape] ||= []).push(r.baselineSignErrorRate);
}
for (const s of Object.keys(byShape)) {
  const mean = byShape[s].reduce((a, b) => a + b, 0) / byShape[s].length;
  say(`  ${s.padEnd(9)} centroid sign error = ${(mean * 100).toFixed(2)}%`);
}
assert.ok(p.baselineSignErrorRate > 0.05, 'centroid baseline must be materially worse');
assert.ok(byShape.torus.reduce((a, b) => a + b, 0) / byShape.torus.length > 0.2, 'torus baseline must fail > 20%');
assert.ok(byShape.sphere.reduce((a, b) => a + b, 0) / byShape.sphere.length < 0.01, 'sphere baseline should be fine (honest: baseline is not weak everywhere)');

say('\n== G-D determinism ==');
const s0 = specs[0];
const h1 = cloudHash(buildCloud(s0));
const h2 = cloudHash(buildCloud(s0));
say(`cloud hash twice: ${h1} ${h2}`);
assert.equal(h1, h2);
const r1 = pipeline(s0);
const r2 = pipeline(s0);
say(`pipeline twice: median ${r1.medianAngleDeg.toFixed(6)} vs ${r2.medianAngleDeg.toFixed(6)}, frustrated ${r1.frustratedEdges} vs ${r2.frustratedEdges}`);
assert.equal(r1.medianAngleDeg, r2.medianAngleDeg);
assert.equal(r1.frustratedEdges, r2.frustratedEdges);

say('\n== G-E kNN vs brute-force oracle (no shared constants) ==');
const small = buildCloud({ id: 'oracle', shape: 'torus', seed: 777, n: 600, noise: 0, params: {} });
const g = knnGraph(small.points, 600, K_DEFAULT);
const qs = Array.from({ length: 60 }, (_, i) => (i * 37) % 600);
const truthK = bruteKnn(small.points, 600, K_DEFAULT, qs);
let mismatch = 0, maxDiff = 0;
for (let qi = 0; qi < qs.length; qi++) {
  const q = qs[qi];
  for (let t = 0; t < K_DEFAULT; t++) {
    const gd = g.dists[q * K_DEFAULT + t];
    const bd = truthK[qi][t][0];
    const diff = Math.abs(gd - bd);
    if (diff > 1e-12 * (small.diag * small.diag)) mismatch++;
    if (diff > maxDiff) maxDiff = diff;
  }
}
say(`queries=${qs.length} k=${K_DEFAULT} mismatches=${mismatch} maxAbsDiff=${maxDiff.toExponential(3)}`);
assert.equal(mismatch, 0, 'grid kNN must match brute force on all sampled queries');

say('\n== G-F scale invariance (1e-6 .. 1e6) ==');
const scales = [1e-6, 1e-3, 1, 1e3, 1e6];
const baseSpec = { id: 'scale', shape: 'torus', seed: 4242, n: 1200, noise: 0, params: {} };
const base = pipeline(baseSpec);
say(`scale        medianAngle   signErr   frustrated  components`);
for (const s of scales) {
  const c = buildCloud({ ...baseSpec, id: 'scale-' + s });
  for (let i = 0; i < c.points.length; i++) c.points[i] *= s;   // truth normals are scale-free
  const knn = knnGraph(c.points, c.n, K_DEFAULT);
  const pca = pcaNormals(c.points, c.n, knn.neighbors, K_DEFAULT);
  const o = orientByMST(c.points, c.n, knn.neighbors, pca, K_DEFAULT);
  const au = parityAudit(c.n, knn.neighbors, o.normals, o.tree, K_DEFAULT);
  const sc = score(o.normals, c.truth, c.n);
  say(`${String(s).padEnd(11)} ${sc.medianAngleDeg.toFixed(9)}   ${(sc.signErrorRate * 100).toFixed(3)}%   ${au.frustrated}           ${o.components}`);
  assert.equal(au.frustrated, base.frustratedEdges, `frustration must not depend on scale ${s}`);
  assert.equal(o.components, base.components, `components must not depend on scale ${s}`);
  assert.ok(Math.abs(sc.medianAngleDeg - base.medianAngleDeg) < 1e-6, `angle must not depend on scale ${s}`);
}

say('\n== G-G rotation invariance ==');
{
  const th = 0.7, ph = 1.1;
  const rz = [[Math.cos(th), -Math.sin(th), 0], [Math.sin(th), Math.cos(th), 0], [0, 0, 1]];
  const ry = [[Math.cos(ph), 0, Math.sin(ph)], [0, 1, 0], [-Math.sin(ph), 0, Math.cos(ph)]];
  const rot = (m, v) => [m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2], m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2], m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2]];
  const spec = { id: 'rot', shape: 'saddle', seed: 909, n: 1200, noise: 0, params: {} };
  const c = buildCloud(spec);
  const ref = pipeline(spec);
  const cp = new Float64Array(c.points.length), ct = new Float64Array(c.truth.length);
  for (let i = 0; i < c.n; i++) {
    const pp = rot(ry, rot(rz, [c.points[3 * i], c.points[3 * i + 1], c.points[3 * i + 2]]));
    const tt = rot(ry, rot(rz, [c.truth[3 * i], c.truth[3 * i + 1], c.truth[3 * i + 2]]));
    for (let k = 0; k < 3; k++) { cp[3 * i + k] = pp[k]; ct[3 * i + k] = tt[k]; }
  }
  const knn = knnGraph(cp, c.n, K_DEFAULT);
  const pca = pcaNormals(cp, c.n, knn.neighbors, K_DEFAULT);
  const o = orientByMST(cp, c.n, knn.neighbors, pca, K_DEFAULT);
  const au = parityAudit(c.n, knn.neighbors, o.normals, o.tree, K_DEFAULT);
  say(`reference frustrated=${ref.frustratedEdges} rotated frustrated=${au.frustrated} components=${o.components}`);
  assert.ok(Math.abs(au.frustrated - ref.frustratedEdges) <= 2, 'rotation must not change frustration materially');
  assert.equal(o.components, 1);
}

say('\n== G-H detector fires on an injected sign flip ==');
{
  const spec = { id: 'inject', shape: 'sphere', seed: 31337, n: 800, noise: 0, params: {} };
  const c = buildCloud(spec);
  const knn = knnGraph(c.points, c.n, K_DEFAULT);
  const pca = pcaNormals(c.points, c.n, knn.neighbors, K_DEFAULT);
  const o = orientByMST(c.points, c.n, knn.neighbors, pca, K_DEFAULT);
  const clean = parityAudit(c.n, knn.neighbors, o.normals, o.tree, K_DEFAULT);
  const flipped = Float64Array.from(o.normals);
  for (let k = 0; k < 3; k++) flipped[3 * 0 + k] *= -1; // flip exactly one point
  const dirty = parityAudit(c.n, knn.neighbors, flipped, o.tree, K_DEFAULT);
  say(`clean frustrated=${clean.frustrated} after one injected flip=${dirty.frustrated}`);
  assert.equal(clean.frustrated, 0);
  assert.ok(dirty.frustrated > 0, 'auditor must see a single injected flip');
}

say('\n== G-I noise sensitivity (2% of spacing, 2 variants per shape) ==');
for (const shape of SHAPES) {
  const rs = results.filter((r) => r.shape === shape && r.noise > 0);
  if (!rs.length) continue;
  const mean = rs.reduce((a, r) => a + r.medianAngleDeg, 0) / rs.length;
  const fr = rs.reduce((a, r) => a + r.frustratedEdges, 0);
  say(`  ${shape.padEnd(9)} median angle=${mean.toFixed(3)} deg frustrated=${fr}`);
}
const noisyOrientable = orientable.filter((r) => r.noise > 0);
assert.ok(noisyOrientable.every((r) => r.frustratedEdges === 0), '2% noise must not fabricate frustration');

say(`\nALL ASSERTIONS PASSED in ${Date.now() - t0} ms`);