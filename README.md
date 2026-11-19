# normal-parity-lab

**Promise: orient the normals of a 120,000-point, 40-cloud corpus with 0 sign errors and 0 frustrated
graph edges (693,730 non-tree edges audited), median error 0.848° vs the analytic surface normal —
while the per-point "centroid-outward" baseline makes 18,145 sign errors (15.12%).**

Upstream: none

A point cloud has no normal signs. PCA gives a normal *line*; something must pick a side. The usual
answer is "point away from the centroid", which silently fails on any surface that folds back toward
its own centroid. This tool propagates signs over a minimum spanning tree of the k-NN graph and then
**audits every remaining (non-tree) edge**: a consistent field must satisfy `dot(n_u, n_v) > 0` on all
of them. The count of edges that cannot be satisfied is a certificate, not a heuristic — and on a
Möbius band it is provably non-zero.

## Run it

```bash
node tools/gen-sample.mjs     # deterministic corpus -> data/corpus.json
node bench/verify.mjs         # 9 assertion groups, exit 0 = pass  (1.4 s)
node bench/bench.mjs          # writes bench/result.json            (4.3 s)
python3 -m http.server 4173    # then open http://localhost:4173
```

No dependencies, no build step, no `npm install`. The browser imports three.js 0.160.0 from a pinned
CDN importmap and calls the **same** `src/core.mjs` that node runs, so the on-screen numbers and the
`bench/result.json` numbers come from one implementation.

## Evidence (`bench/result.json`, this machine, node v24.19.0)

| what | measured |
| --- | --- |
| orientable corpus | 40 clouds / 120,000 points, 5 shape families |
| frustrated non-tree edges | **0 / 693,730** (rate 0.0000%) |
| sign errors after one global flip | **0 / 120,000** |
| angle error | median **0.848°**, mean 1.790°, p95 5.039° |
| baseline (per-point centroid-outward) | **18,145 / 120,000 sign errors = 15.12%** (torus 27.25%, saddle 48.35%) |
| Möbius negative control | **672 / 140,237 frustrated (0.479%)**, every one of 8 clouds > 0 (40–134) |
| injected single flip | 0 → **11** frustrated edges |
| k-NN vs brute force (1,200 pts) | **4.106 ms vs 349.565 ms = 85.1x**, rsd 1.1% |
| pipeline, 3,000 pts | cold **25.55 ms**, median **20.01 ms** (14.82 box – 24.51 Möbius), rsd 1.0–2.3% |
| scale sweep 1e-6 … 1e6 | identical frustration, components and median angle (9 decimals) |

The oracle in `G-E` is an O(n²) brute-force k-NN that shares no code, no grid and no epsilon with the
grid index. `G-D` rebuilds the cloud and re-runs the pipeline and compares bit-for-bit.

## Known limits (read before reusing)

- **Global sign is not recoverable from geometry alone.** Scoring resolves the single global flip by
  majority vote against truth; that is the only truth-dependent step, and it is applied identically to
  the baseline. A real scan has no "outward".
- **Sharp features are averaged away.** On the box, median error is 0.000° but p95 is 26.75° and max is
  54.06°: samples within one k-NN radius of an edge get a PCA normal that mixes two faces. The parity
  audit still reports 0, because it only checks *relative* sign.
- **The noise column is finite.** 2% of the mean spacing is tolerated (median error rises to ~1–2.5°,
  frustration stays 0). Above that, this implementation is untested — no claim is made.
- **Density matters.** `cell = bboxDiag * sqrt(k/n)` assumes roughly uniform surface sampling; a cloud
  with 100x density variation will silently over/under-connect. No adaptive density estimate.
- **No quantization, no outlier rejection, no sharp-edge reconstruction.** Integer-exactness is not
  claimed either: all thresholds are relative to the bounding-box diagonal (there is no absolute
  epsilon), and the scale sweep is the evidence for that.
- The corpus is procedural (6 analytic families), not a real scanner dataset. Möbius is the only
  non-orientable case.

## 本轮创新

**B 档：把"法向定向"从启发式变成可判定的证书。** 现有工具（含本账号此前的 `pointcloud-studio`、
`pointpick-lab`）只做 k-NN 查询或拾取，没有一条把「定向是否全局一致」写成可复算的数字。这里的新东西
是 **非树边奇偶审计**：MST 只能保证生成树内一致，剩下的每一条非树边都是独立检验；在 5 类可定向曲面
上 69 万条边全部通过，在 Möbius 带上必然出现 672 条不可满足的边（0.479%），并且单点注入翻转会被
立即检出（0→11）。以前做不到的原因很直接：没有把"图上的符号一致性"当成一等对象来度量，只报告角度误差。

## Files

```
src/core.mjs          pure ES module: sampling, grid k-NN, PCA, MST, parity, scoring
tools/gen-sample.mjs  deterministic corpus generator (48 specs, 144,000 points)
bench/verify.mjs      executable assertions (9 groups)
bench/bench.mjs       timings + aggregates -> bench/result.json
index.html, main.js   three.js viewer: point colour = sign agreement, sticks = normals
baseline/centroid.mjs standalone baseline (same sign rule, no propagation)
```

MIT licensed. © 2026 llinok111.