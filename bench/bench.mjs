// bench/bench.mjs — real timings and corpus aggregates written to bench/result.json.
// Cold start = first pipeline call; then 1 warmup + 5 measured samples per shape.
import { writeFileSync } from 'node:fs';
import {
  makeCorpus, buildCloud, pipeline, pool, knnGraph, bruteKnn, K_DEFAULT, SHAPES, nowMs,
} from '../src/core.mjs';

const t0 = nowMs();
const stats = (a) => {
  const s = [...a].sort((x, y) => x - y);
  const mean = s.reduce((x, y) => x + y, 0) / s.length;
  const variance = s.reduce((x, y) => x + (y - mean) ** 2, 0) / s.length;
  return {
    samples: s.length, min: s[0], median: s[Math.floor((s.length - 1) / 2)],
    p95: s[Math.min(s.length - 1, Math.floor(0.95 * (s.length - 1)))],
    max: s[s.length - 1], mean, stddev: Math.sqrt(variance),
    rsd: mean ? Math.sqrt(variance) / mean : 0,
  };
};

// ---------------------------------------------------------------- corpus aggregate
const specs = makeCorpus();
const results = specs.map((s) => pipeline(s));
const orientable = results.filter((r) => r.shape !== 'mobius');
const mobius = results.filter((r) => r.shape === 'mobius');
const p = pool(orientable);
const mob = pool(mobius);
const perShape = SHAPES.map((shape) => {
  const rs = results.filter((r) => r.shape === shape);
  const agg = pool(rs);
  return {
    shape, clouds: rs.length, points: agg.points,
    medianAngleDeg: agg.medianAngleDeg, p95AngleDeg: agg.p95AngleDeg, maxAngleDeg: agg.maxAngleDeg,
    nonTreeEdges: agg.nonTreeEdges, frustratedEdges: agg.frustratedEdges,
    signErrorRate: agg.signErrorRate, baselineSignErrorRate: agg.baselineSignErrorRate,
  };
});

// ---------------------------------------------------------------- kNN speed vs brute force
const knnCloud = buildCloud({ id: 'knn-speed', shape: 'torus', seed: 2026, n: 1200, noise: 0, params: {} });
const gridTimes = [], bruteTimes = [];
for (let rep = 0; rep < 6; rep++) {
  let a = nowMs();
  knnGraph(knnCloud.points, knnCloud.n, K_DEFAULT);
  gridTimes.push(nowMs() - a);
  a = nowMs();
  bruteKnn(knnCloud.points, knnCloud.n, K_DEFAULT);
  bruteTimes.push(nowMs() - a);
}
const gridStats = stats(gridTimes.slice(1));
const bruteStats = stats(bruteTimes.slice(1));

// ---------------------------------------------------------------- cold start + samples
const warmSpecs = SHAPES.map((shape) => ({ id: 'bench-' + shape, shape, seed: 5150, n: 3000, noise: 0, params: {} }));
const coldStartMs = nowMs();
pipeline(warmSpecs[0]);
const cold = nowMs() - coldStartMs;
const timings = {};
for (const spec of warmSpecs) {
  pipeline(spec); // warmup
  const samples = [];
  for (let rep = 0; rep < 5; rep++) {
    const a = nowMs();
    pipeline(spec);
    samples.push(nowMs() - a);
  }
  timings[spec.shape] = { points: spec.n, coldStartMs: cold, ...stats(samples) };
}
const allMedians = SHAPES.map((s) => timings[s].median);
const totalTiming = stats(allMedians);

const out = {
  generated_at: new Date().toISOString(),
  node: process.version,
  platform: process.platform + '/' + process.arch,
  method: {
    k: K_DEFAULT,
    cell: 'bboxDiag * sqrt(k/n), radius-bounded ring search with exact cell de-collision',
    stop_rule: '(ring*cell)^2 >= kth_distance (cells outside have min distance ring*cell)',
    orient: 'Prim MST over kNN graph, weight = 1 - |dot|; seed sign = away from centroid',
    certificate: 'every non-tree edge must satisfy dot > 0; otherwise the edge is frustrated',
    inner_loop_reps: 5,
    warmup: 1,
  },
  corpus: {
    clouds: specs.length, points: specs.reduce((a, s) => a + s.n, 0),
    noisyClouds: specs.filter((s) => s.noise > 0).length,
    orientable: {
      clouds: p.clouds, points: p.points,
      nonTreeEdges: p.nonTreeEdges, frustratedEdges: p.frustratedEdges, frustrationRate: p.frustrationRate,
      signErrorPoints: p.signErrorPoints, signErrorRate: p.signErrorRate,
      medianAngleDeg: p.medianAngleDeg, meanAngleDeg: p.meanAngleDeg,
      p95AngleDeg: p.p95AngleDeg, maxAngleDeg: p.maxAngleDeg,
    },
    mobius: {
      clouds: mob.clouds, points: mob.points,
      nonTreeEdges: mob.nonTreeEdges, frustratedEdges: mob.frustratedEdges, frustrationRate: mob.frustrationRate,
      perCloudFrustrated: mobius.map((r) => r.frustratedEdges),
    },
    baseline: {
      signErrorPoints: p.baselineSignErrorPoints, signErrorRate: p.baselineSignErrorRate,
      medianAngleDeg: (() => {
        const all = [];
        for (const r of orientable) for (const a of r.baselineAngles) all.push(a);
        all.sort((x, y) => x - y);
        return all[Math.floor((all.length - 1) / 2)];
      })(),
    },
    perShape,
  },
  knn_vs_brute: {
    points: knnCloud.n, queries: knnCloud.n, k: K_DEFAULT,
    grid: gridStats, brute: bruteStats, speedup: bruteStats.median / gridStats.median,
  },
  timing: { coldStartMs: cold, per_shape: timings, median_of_shape_medians: totalTiming.median, total_bench_ms: nowMs() - t0 },
};
writeFileSync(new URL('result.json', import.meta.url), JSON.stringify(out, null, 1) + '\n');
console.log(JSON.stringify(out, null, 1));