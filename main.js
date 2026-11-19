// main.js — rendering only. Every number on screen is computed by src/core.mjs,
// the same module bench/verify.mjs executes in node.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import {
  SHAPES, K_DEFAULT, buildCloud, knnGraph, pcaNormals, orientByMST,
  parityAudit, centroidOutward, score,
} from './src/core.mjs';

const $ = (id) => document.getElementById(id);
const view = $('view');

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(2, window.devicePixelRatio));
renderer.setSize(view.clientWidth, view.clientHeight);
view.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b0e14);
const camera = new THREE.PerspectiveCamera(45, view.clientWidth / view.clientHeight, 0.01, 1000);
camera.position.set(2.2, 1.7, 2.6);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
scene.add(new THREE.AmbientLight(0xffffff, 1.1));

const cloudGroup = new THREE.Group();
scene.add(cloudGroup);

let corpus = null;
let state = { shape: 'torus', variant: 0, useBaseline: false, showSticks: true };
let cache = null;

function clearGroup() {
  for (const c of [...cloudGroup.children]) {
    cloudGroup.remove(c);
    c.geometry?.dispose();
    c.material?.dispose();
  }
}

function buildSpec() {
  const spec = corpus.specs.find((s) => s.shape === state.shape && s.id.endsWith('-' + String(state.variant).padStart(2, '0')));
  return spec || corpus.specs.find((s) => s.shape === state.shape);
}

function run() {
  const spec = buildSpec();
  const cloud = buildCloud(spec);
  const { points, truth, n } = cloud;
  const knn = knnGraph(points, n, K_DEFAULT);
  const pca = pcaNormals(points, n, knn.neighbors, K_DEFAULT);
  const oriented = orientByMST(points, n, knn.neighbors, pca, K_DEFAULT);
  const parity = parityAudit(n, knn.neighbors, oriented.normals, oriented.tree, K_DEFAULT);
  const base = centroidOutward(points, n, pca);
  const shown = state.useBaseline ? base : oriented.normals;
  const own = score(shown, truth, n);
  const ref = score(oriented.normals, truth, n);
  // display-only global flip (the algorithm itself never sees truth)
  const flip = own.signErrorRate > 0.5 ? -1 : 1;

  $('noise').textContent = ((spec.noise || 0) * 100).toFixed(0) + '%';
  $('m-med').textContent = own.medianAngleDeg.toFixed(3) + '°';
  $('m-p95').textContent = own.p95AngleDeg.toFixed(2) + '°';
  $('m-sign').textContent = (own.signErrorRate * 100).toFixed(2) + '%';
  $('m-fru').textContent = parity.frustrated + ' / ' + parity.nonTree;
  $('m-nt').textContent = String(parity.edges);
  $('m-comp').textContent = String(oriented.components);
  $('m-fru').className = parity.frustrated > 0 ? 'bad' : 'ok';
  $('m-sign').className = own.signErrorRate > 0.05 ? 'bad' : 'ok';

  clearGroup();
  const pos = new Float32Array(n * 3);
  const col = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    pos[3 * i] = points[3 * i]; pos[3 * i + 1] = points[3 * i + 1]; pos[3 * i + 2] = points[3 * i + 2];
    const d = (shown[3 * i] * truth[3 * i] + shown[3 * i + 1] * truth[3 * i + 1] + shown[3 * i + 2] * truth[3 * i + 2]) * flip;
    // green = agrees, red = flipped, brightness follows angle (0 deg = bright)
    const t = Math.min(1, own.angles[i] / 30);
    if (d >= 0) col.set([0.30 + 0.35 * t, 0.92 - 0.35 * t, 0.62 - 0.25 * t], 3 * i);
    else col.set([1.0, 0.28 + 0.3 * t, 0.32], 3 * i);
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
  cloudGroup.add(new THREE.Points(geo, new THREE.PointsMaterial({ size: 0.016, vertexColors: true, sizeAttenuation: true })));

  if (state.showSticks) {
    const step = Math.max(1, Math.floor(n / 700));
    const verts = [];
    for (let i = 0; i < n; i += step) {
      const L = cloud.diag * 0.018;
      verts.push(
        points[3 * i], points[3 * i + 1], points[3 * i + 2],
        points[3 * i] + shown[3 * i] * L * flip, points[3 * i + 1] + shown[3 * i + 1] * L * flip, points[3 * i + 2] + shown[3 * i + 2] * L * flip,
      );
    }
    const lg = new THREE.BufferGeometry();
    lg.setAttribute('position', new THREE.Float32BufferAttribute(verts, 3));
    cloudGroup.add(new THREE.LineSegments(lg, new THREE.LineBasicMaterial({ color: 0x9fd0ff, transparent: true, opacity: 0.55 })));
  }

  const box = new THREE.Box3().setFromBufferAttribute(geo.getAttribute('position'));
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3()).length();
  geo.translate(-center.x, -center.y, -center.z);
  controls.target.set(0, 0, 0);
  camera.position.set(size * 0.85, size * 0.6, size * 0.95);
  camera.near = size * 0.01;
  camera.far = size * 20;
  camera.updateProjectionMatrix();
  cache = { parity, own, ref, spec, orientation: shown };
}

function fit() {
  const w = view.clientWidth, h = view.clientHeight;
  renderer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
window.addEventListener('resize', fit);
fit();

function loop() {
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(loop);
}

async function boot() {
  corpus = await (await fetch('./data/corpus.json')).json();
  const shapeSel = $('shape'), varSel = $('variant');
  for (const s of SHAPES) {
    const o = document.createElement('option');
    o.value = s; o.textContent = s;
    shapeSel.appendChild(o);
  }
  shapeSel.value = state.shape;
  const fillVariants = () => {
    varSel.innerHTML = '';
    for (let i = 0; i < 8; i++) {
      const o = document.createElement('option');
      o.value = String(i);
      o.textContent = 'variant ' + String(i).padStart(2, '0') + (i >= 6 ? ' (2% noise)' : '');
      varSel.appendChild(o);
    }
    varSel.value = '0';
  };
  fillVariants();
  shapeSel.onchange = () => { state.shape = shapeSel.value; state.variant = 0; fillVariants(); run(); };
  varSel.onchange = () => { state.variant = Number(varSel.value); run(); };
  $('btn-mst').onclick = () => {
    state.useBaseline = false;
    $('btn-mst').setAttribute('aria-pressed', 'true');
    $('btn-base').setAttribute('aria-pressed', 'false');
    run();
  };
  $('btn-base').onclick = () => {
    state.useBaseline = true;
    $('btn-mst').setAttribute('aria-pressed', 'false');
    $('btn-base').setAttribute('aria-pressed', 'true');
    run();
  };
  $('btn-lines').onclick = (e) => {
    state.showSticks = !state.showSticks;
    e.currentTarget.setAttribute('aria-pressed', String(state.showSticks));
    run();
  };
  run();
  loop();
}

boot().catch((err) => {
  $('node').textContent = 'error: ' + err.message;
  console.error(err);
});

// Cross-check the node-recorded run against this browser session.
fetch('./bench/result.json').then((r) => r.json()).then((res) => {
  const c = res.corpus;
  $('node-corpus').textContent = c.orientable.clouds + ' / ' + c.orientable.points.toLocaleString();
  $('node-fru').textContent = c.orientable.frustratedEdges + ' / ' + c.orientable.nonTreeEdges;
  $('node-sign').textContent = (c.orientable.signErrorRate * 100).toFixed(3) + '%';
  $('node-base').textContent = (c.baseline.signErrorRate * 100).toFixed(2) + '%';
  $('node-mob').textContent = c.mobius.frustratedEdges + ' / ' + c.mobius.nonTreeEdges;
  $('node-time').textContent = res.timing.median_of_shape_medians.toFixed(2) + ' ms / cloud';
  $('node').textContent = 'node ' + res.node + ' · ' + res.timing.per_shape[state.shape]?.points + ' pts/cloud · k=' + res.method.k;
}).catch(() => { $('node').textContent = 'bench/result.json not reachable (serve from repo root)'; });