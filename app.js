(function () {
  'use strict';

  const HAND_MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';
  const WASM_PATH = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm';
  const NUM_PARTICLES = 8000;
  const TEMPLATE_NAMES = ['Hearts', 'Flowers', 'Saturn', 'Fireworks', 'Stars'];

  let scene, camera, renderer, points, geometry;
  let positionAttr, colorAttr;
  let handLandmarker = null;
  let video = document.getElementById('video');
  let statusEl = document.getElementById('status');
  let templateNameEl = document.getElementById('template-name');

  // Gesture state (smoothed)
  let targetExpansion = 1, expansion = 1;
  let targetHue = 0.95, hue = 0.95;
  let targetTemplateIndex = 0, templateIndex = 0;
  let templateBlend = 0; // 0 = current template, 1 = next template

  // Precomputed template data: { positions: Float32Array(3*N), colors: Float32Array(3*N) }
  let templateData = [];

  // --- Particle template generators ---
  function hslToRgb(h, s, l) {
    let r, g, b;
    if (s === 0) { r = g = b = l; } else {
      const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
      const p = 2 * l - q;
      r = hue2rgb(p, q, h + 1/3);
      g = hue2rgb(p, q, h);
      b = hue2rgb(p, q, h - 1/3);
    }
    return [r, g, b];
  }
  function hue2rgb(p, q, t) {
    if (t < 0) t += 1; if (t > 1) t -= 1;
    if (t < 1/6) return p + (q - p) * 6 * t;
    if (t < 1/2) return q;
    return t < 2/3 ? p + (q - p) * (2/3 - t) * 6 : p;
  }

  function buildHeart(n) {
    const positions = [], colors = [];
    const scale = 0.35;
    for (let i = 0; i < n; i++) {
      const t = (i / n) * Math.PI * 2 * 4;
      const x = scale * 16 * Math.pow(Math.sin(t), 3);
      const y = scale * (13 * Math.cos(t) - 5 * Math.cos(2*t) - 2 * Math.cos(3*t) - Math.cos(4*t));
      const z = (Math.random() - 0.5) * scale * 4;
      positions.push(x * 0.08, y * 0.08, z);
      const rgb = hslToRgb(0.95 + (i / n) * 0.05, 0.8, 0.7);
      colors.push(rgb[0], rgb[1], rgb[2]);
    }
    return { positions: new Float32Array(positions), colors: new Float32Array(colors) };
  }

  function buildFlower(n) {
    const positions = [], colors = [];
    const petals = 8;
    const rings = 5;
    let idx = 0;
    for (let ring = 0; ring < rings && idx < n; ring++) {
      const r0 = 0.05 + ring * 0.12;
      const pts = Math.max(20, Math.floor((n / rings) / petals));
      for (let p = 0; p < petals && idx < n; p++) {
        const a0 = (p / petals) * Math.PI * 2;
        for (let k = 0; k < pts && idx < n; k++) {
          const t = k / pts;
          const a = a0 + t * (Math.PI / petals);
          const r = r0 * (0.7 + 0.3 * Math.sin(t * Math.PI));
          const x = r * Math.cos(a);
          const z = r * Math.sin(a);
          const y = (Math.random() - 0.5) * 0.05;
          positions.push(x, y, z);
          const rgb = hslToRgb(0.08 + (ring / rings) * 0.1, 0.85, 0.6);
          colors.push(rgb[0], rgb[1], rgb[2]);
          idx++;
        }
      }
    }
    while (idx < n) {
      positions.push((Math.random() - 0.5) * 0.3, (Math.random() - 0.5) * 0.1, (Math.random() - 0.5) * 0.3);
      const rgb = hslToRgb(0.1, 0.8, 0.65);
      colors.push(rgb[0], rgb[1], rgb[2]);
      idx++;
    }
    return { positions: new Float32Array(positions.slice(0, n * 3)), colors: new Float32Array(colors.slice(0, n * 3)) };
  }

  function buildSaturn(n) {
    const positions = [], colors = [];
    const sphereN = Math.floor(n * 0.4);
    const ringN = n - sphereN;
    for (let i = 0; i < sphereN; i++) {
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);
      const r = 0.15 * (0.85 + Math.random() * 0.3);
      const x = r * Math.sin(phi) * Math.cos(theta);
      const y = r * Math.sin(phi) * Math.sin(theta);
      const z = r * Math.cos(phi);
      positions.push(x, y, z);
      const rgb = hslToRgb(0.12, 0.7, 0.75);
      colors.push(rgb[0], rgb[1], rgb[2]);
    }
    const ringR0 = 0.22, ringR1 = 0.38;
    for (let i = 0; i < ringN; i++) {
      const a = (i / ringN) * Math.PI * 2 * 3;
      const r = ringR0 + (ringR1 - ringR0) * Math.random();
      const x = r * Math.cos(a);
      const z = r * Math.sin(a);
      const y = (Math.random() - 0.5) * 0.02;
      positions.push(x, y, z);
      const rgb = hslToRgb(0.14, 0.6, 0.8);
      colors.push(rgb[0], rgb[1], rgb[2]);
    }
    return { positions: new Float32Array(positions), colors: new Float32Array(colors) };
  }

  function buildFireworks(n) {
    const positions = [], colors = [];
    const bursts = 5;
    const perBurst = Math.floor(n / bursts);
    const hues = [0, 0.05, 0.55, 0.95, 0.7];
    for (let b = 0; b < bursts; b++) {
      const ox = (Math.random() - 0.5) * 0.4;
      const oy = (Math.random() - 0.5) * 0.2;
      const oz = (Math.random() - 0.5) * 0.4;
      const hue = hues[b];
      for (let i = 0; i < perBurst; i++) {
        const theta = Math.random() * Math.PI * 2;
        const phi = Math.acos(2 * Math.random() - 1);
        const speed = 0.08 + Math.random() * 0.15;
        const x = ox + speed * Math.sin(phi) * Math.cos(theta);
        const y = oy + speed * Math.sin(phi) * Math.sin(theta);
        const z = oz + speed * Math.cos(phi);
        positions.push(x, y, z);
        const rgb = hslToRgb(hue, 0.9, 0.5 + Math.random() * 0.4);
        colors.push(rgb[0], rgb[1], rgb[2]);
      }
    }
    const used = bursts * perBurst;
    for (let i = used; i < n; i++) {
      positions.push((Math.random() - 0.5) * 0.5, (Math.random() - 0.5) * 0.5, (Math.random() - 0.5) * 0.5);
      colors.push(1, 0.9, 0.3);
    }
    return { positions: new Float32Array(positions), colors: new Float32Array(colors) };
  }

  function buildStars(n) {
    const positions = [], colors = [];
    for (let i = 0; i < n; i++) {
      const x = (Math.random() - 0.5) * 1.2;
      const y = (Math.random() - 0.5) * 1.2;
      const z = (Math.random() - 0.5) * 1.2;
      positions.push(x, y, z);
      const warm = Math.random() > 0.5;
      const h = warm ? 0.1 + Math.random() * 0.05 : 0.55 + Math.random() * 0.05;
      const rgb = hslToRgb(h, 0.3, 0.85 + Math.random() * 0.15);
      colors.push(rgb[0], rgb[1], rgb[2]);
    }
    return { positions: new Float32Array(positions), colors: new Float32Array(colors) };
  }

  function buildTemplates() {
    templateData = [
      buildHeart(NUM_PARTICLES),
      buildFlower(NUM_PARTICLES),
      buildSaturn(NUM_PARTICLES),
      buildFireworks(NUM_PARTICLES),
      buildStars(NUM_PARTICLES)
    ];
  }

  // --- Three.js setup ---
  function initThree() {
    const canvas = document.getElementById('canvas');
    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 100);
    camera.position.set(0, 0, 1.8);
    renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

    geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(NUM_PARTICLES * 3);
    const colors = new Float32Array(NUM_PARTICLES * 3);
    const sizes = new Float32Array(NUM_PARTICLES);
    for (let i = 0; i < NUM_PARTICLES; i++) sizes[i] = 0.8 + Math.random() * 0.4;
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    geometry.setAttribute('size', new THREE.BufferAttribute(sizes, 1));

    const mat = new THREE.PointsMaterial({
      size: 0.012,
      vertexColors: true,
      transparent: true,
      opacity: 0.9,
      sizeAttenuation: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false
    });
    points = new THREE.Points(geometry, mat);
    scene.add(points);

    positionAttr = geometry.attributes.position;
    colorAttr = geometry.attributes.color;

    buildTemplates();
    // Seed with first template
    positionAttr.array.set(templateData[0].positions);
    colorAttr.array.set(templateData[0].colors);
    positionAttr.needsUpdate = true;
    colorAttr.needsUpdate = true;

    window.addEventListener('resize', () => {
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    });
  }

  // --- Hand gesture → parameters ---
  function countExtendedFingers(landmarks) {
    const fingerTips = [4, 8, 12, 16, 20];
    const fingerPips = [3, 6, 10, 14, 18];
    let count = 0;
    for (let i = 0; i < 5; i++) {
      const tip = landmarks[fingerTips[i]];
      const pip = landmarks[fingerPips[i]];
      if (tip.y < pip.y) count++;
    }
    return count;
  }

  function palmSpread(landmarks) {
    const wrist = landmarks[0];
    const middle = landmarks[9];
    const index = landmarks[5];
    const ring = landmarks[13];
    const d1 = Math.hypot(middle.x - wrist.x, middle.y - wrist.y);
    const d2 = Math.hypot(index.x - ring.x, index.y - ring.y);
    return Math.min(2, (d1 + d2 * 2) * 2);
  }

  function updateGestureFromHand(landmarks) {
    const fingers = countExtendedFingers(landmarks);
    targetTemplateIndex = Math.min(4, Math.max(0, fingers));
    targetExpansion = 0.3 + palmSpread(landmarks) * 0.5;
    const palmY = (landmarks[0].y + landmarks[5].y + landmarks[9].y) / 3;
    targetHue = palmY;
  }

  function noHand() {
    targetExpansion += (1 - targetExpansion) * 0.02;
    targetHue += (0.95 - targetHue) * 0.02;
  }

  // --- Hand tracking init ---
  async function initHandTracking() {
    try {
      statusEl.textContent = 'Loading hand model…';
      const vision = await globalThis.FilesetResolver.forVisionTasks(WASM_PATH);
      handLandmarker = await globalThis.HandLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: HAND_MODEL_URL },
        numHands: 1,
        runningMode: 'VIDEO'
      });
      statusEl.textContent = 'Starting camera…';
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, facingMode: 'user' } });
      video.srcObject = stream;
      await video.play();
      statusEl.textContent = 'Show your hand to control particles';
    } catch (e) {
      statusEl.textContent = 'Camera/hand error: ' + (e.message || e);
      console.error(e);
    }
  }

  let lastVideoTime = -1;
  function detectHands() {
    if (!handLandmarker || video.readyState < 2) return;
    if (video.currentTime === lastVideoTime) return;
    lastVideoTime = video.currentTime;
    try {
      const result = handLandmarker.detectForVideo(video, performance.now() / 1000);
      if (result.landmarks && result.landmarks.length > 0) {
        updateGestureFromHand(result.landmarks[0]);
      } else {
        noHand();
      }
    } catch (_) { noHand(); }
  }

  // --- Animation ---
  function animate(time) {
    requestAnimationFrame(animate);
    detectHands();

    expansion += (targetExpansion - expansion) * 0.08;
    hue += (targetHue - hue) * 0.06;
    if (targetTemplateIndex !== templateIndex) {
      templateBlend += 0.06;
      if (templateBlend >= 1) {
        templateBlend = 0;
        templateIndex = targetTemplateIndex;
        templateNameEl.textContent = TEMPLATE_NAMES[templateIndex];
      }
    }

    const curr = templateData[templateIndex];
    const next = templateBlend > 0 ? templateData[targetTemplateIndex] : curr;
    const pos = positionAttr.array;
    const col = colorAttr.array;
    const hueShift = (hue - 0.5) * 0.3;

    for (let i = 0; i < NUM_PARTICLES; i++) {
      const i3 = i * 3;
      const px = curr.positions[i3];
      const py = curr.positions[i3 + 1];
      const pz = curr.positions[i3 + 2];
      let nx = px, ny = py, nz = pz;
      if (templateBlend > 0 && next) {
        const t = templateBlend;
        nx = px + t * (next.positions[i3] - px);
        ny = py + t * (next.positions[i3 + 1] - py);
        nz = pz + t * (next.positions[i3 + 2] - pz);
      }
      pos[i3] = nx * expansion;
      pos[i3 + 1] = ny * expansion;
      pos[i3 + 2] = nz * expansion;

      const r = curr.colors[i3], g = curr.colors[i3 + 1], b = curr.colors[i3 + 2];
      let rr = r, gg = g, bb = b;
      if (templateBlend > 0 && next) {
        const t = templateBlend;
        rr = r + t * (next.colors[i3] - r);
        gg = g + t * (next.colors[i3 + 1] - g);
        bb = b + t * (next.colors[i3 + 2] - b);
      }
      const rgb = hslToRgb((hue + hueShift + (i % 100) * 0.0001) % 1, 0.7, 0.7);
      const blend = 0.4;
      col[i3] = rr * (1 - blend) + rgb[0] * blend;
      col[i3 + 1] = gg * (1 - blend) + rgb[1] * blend;
      col[i3 + 2] = bb * (1 - blend) + rgb[2] * blend;
    }
    positionAttr.needsUpdate = true;
    colorAttr.needsUpdate = true;

    points.rotation.y = time * 0.00015;
    renderer.render(scene, camera);
  }

  // --- Keyboard fallback (when no camera or MediaPipe) ---
  function setupKeyboard() {
    document.addEventListener('keydown', function (e) {
      if (e.key === 'ArrowUp') targetExpansion = Math.min(2, targetExpansion + 0.15);
      if (e.key === 'ArrowDown') targetExpansion = Math.max(0.3, targetExpansion - 0.15);
      if (e.key === '1') targetTemplateIndex = 0;
      if (e.key === '2') targetTemplateIndex = 1;
      if (e.key === '3') targetTemplateIndex = 2;
      if (e.key === '4') targetTemplateIndex = 3;
      if (e.key === '5') targetTemplateIndex = 4;
      if (e.key === 'q') targetHue = Math.max(0, targetHue - 0.05);
      if (e.key === 'w') targetHue = Math.min(1, targetHue + 0.05);
    });
  }

  // --- Start ---
  initThree();
  setupKeyboard();
  templateNameEl.textContent = TEMPLATE_NAMES[0];

  function startLoop() {
    requestAnimationFrame(animate);
  }

  if (typeof globalThis.FilesetResolver !== 'undefined' && typeof globalThis.HandLandmarker !== 'undefined') {
    initHandTracking().then(startLoop);
  } else {
    statusEl.textContent = 'Loading MediaPipe… (use keys 1–5, Arrows, Q/W if no camera)';
    const check = setInterval(function () {
      if (typeof globalThis.FilesetResolver !== 'undefined' && typeof globalThis.HandLandmarker !== 'undefined') {
        clearInterval(check);
        initHandTracking().then(startLoop);
      }
    }, 100);
    startLoop();
  }
})();
