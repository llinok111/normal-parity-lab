// tools/gen-sample.mjs — deterministic corpus generator.
// Writes data/corpus.json (specs + per-cloud fingerprints) and prints the file hash.
// Same command twice must produce byte-identical output.
import { writeFileSync, mkdirSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { makeCorpus, buildCloud, cloudHash, K_DEFAULT, SHAPES } from '../src/core.mjs';

const specs = makeCorpus();
const clouds = specs.map((s) => {
  const c = buildCloud(s);
  return { id: s.id, shape: s.shape, n: c.n, hash: cloudHash(c), spacing: c.spacing, diag: c.diag };
});
const doc = {
  version: 1,
  generator: 'tools/gen-sample.mjs',
  k: K_DEFAULT,
  shapes: SHAPES,
  specs,
  clouds,
};
const text = JSON.stringify(doc, null, 1) + '\n';
mkdirSync(new URL('../data/', import.meta.url), { recursive: true });
writeFileSync(new URL('../data/corpus.json', import.meta.url), text);
const sha = createHash('sha256').update(text).digest('hex');
console.log('clouds      :', specs.length);
console.log('points      :', specs.reduce((a, s) => a + s.n, 0));
console.log('noisy specs :', specs.filter((s) => s.noise > 0).length);
for (const shape of SHAPES) {
  const c = clouds.filter((x) => x.shape === shape);
  console.log(String(shape).padEnd(9), c.length, 'clouds', c.map((x) => x.hash.slice(0, 8)).join(' '));
}
console.log('sha256      :', sha);