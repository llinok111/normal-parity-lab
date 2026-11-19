// baseline/centroid.mjs — the naive orientation rule, isolated so it can be run alone.
// Each point independently: if dot(normal, point - centroid) < 0 then flip.
import { buildCloud, knnGraph, pcaNormals, centroidOutward, score, K_DEFAULT } from '../src/core.mjs';

const spec = { id: 'baseline', shape: 'torus', seed: 12345, n: 3000, noise: 0, params: {} };
const cloud = buildCloud(spec);
const knn = knnGraph(cloud.points, cloud.n, K_DEFAULT);
const pca = pcaNormals(cloud.points, cloud.n, knn.neighbors, K_DEFAULT);
const oriented = centroidOutward(cloud.points, cloud.n, pca);
const s = score(oriented, cloud.truth, cloud.n);
console.log(JSON.stringify({
  shape: spec.shape, points: cloud.n,
  medianAngleDeg: s.medianAngleDeg, p95AngleDeg: s.p95AngleDeg,
  signErrorRate: s.signErrorRate, signErrorPoints: Math.round(s.signErrorRate * cloud.n),
}, null, 1));