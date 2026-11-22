/**
 * LLM Pipeline Explorer — Core Interactive Script
 * Features:
 * 1. 3D WebGL Three.js Liquid Glass Stage Showcase (Matches Reference Design)
 * 2. Complete Subword Tokenizer & BPE Compression Slider
 * 3. 64-D Trigram Semantic Embeddings + 3D PCA Power-Iteration with 3D Manifold Warping (Flat / Hyperbolic / Sphere)
 * 4. Multi-Head Self-Attention Q-K-V Matrix & Token Coreference Heatmap
 * 5. Autoregressive Next-Token Decoder with Softmax Temperature & Top-K Sampling
 */

// Initialize Lucide icons
if (window.lucide) {
  lucide.createIcons();
}

// Global Pipeline State
const state = {
  currentInputText: "The cat sat on the mat.",
  vocab: null,
  sortedWords: [],
  wordCounts: {},
  subwordPieces: [],
  COMMON_ENDINGS: ["##s", "##es", "##ed", "##ing", "##er", "##ly", "##y"],
  userText: null,
  manifoldType: "flat", // 'flat' | 'curved' | 'sphere'
  bendFactor: 0.0,
  activeAttentionHead: 0,
  activeStageStep: 0,
  temperature: 0.7,
  topK: 5
};

// =====================================================================
// PART 1: 3D THREE.JS LIQUID GLASS PIPELINE SHOWCASE
// =====================================================================

let scene, camera, renderer, stageCards = [], energyParticles;
let isThreeInitialized = false;

function initThreePipeline() {
  const container = document.getElementById("three-container");
  if (!container) return;

  const width = container.clientWidth || 600;
  const height = container.clientHeight || 380;

  scene = new THREE.Scene();
  scene.fog = new THREE.FogExp2(0x0a0f1c, 0.035);

  camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
  camera.position.set(0, 1.5, 9.5);

  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setSize(width, height);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.2;
  container.innerHTML = "";
  container.appendChild(renderer.domElement);

  // Lighting
  const ambientLight = new THREE.AmbientLight(0xffffff, 0.85);
  scene.add(ambientLight);

  const mainDirLight = new THREE.DirectionalLight(0x38bdf8, 2.5);
  mainDirLight.position.set(5, 8, 5);
  scene.add(mainDirLight);

  const fillPurpleLight = new THREE.PointLight(0xa855f7, 3, 20);
  fillPurpleLight.position.set(-6, 2, 2);
  scene.add(fillPurpleLight);

  const fillPinkLight = new THREE.PointLight(0xec4899, 2.5, 20);
  fillPinkLight.position.set(6, -2, 3);
  scene.add(fillPinkLight);

  // Create 5 Glass Cards representing the 5 Pipeline Stages
  const stageData = [
    { name: "Input", color: 0x38bdf8, subtext: "Raw Prompt" },
    { name: "Tokenization", color: 0x0ea5e9, subtext: "Subwords & IDs" },
    { name: "Embedding", color: 0x8b5cf6, subtext: "64-D Vectors" },
    { name: "Transformer", color: 0xd946ef, subtext: "Multi-Head Attn" },
    { name: "Output", color: 0xf59e0b, subtext: "Logits & Next Token" }
  ];

  const spacing = 2.4;
  const startX = -((stageData.length - 1) * spacing) / 2;

  stageCards = [];
  const cardGroup = new THREE.Group();

  stageData.forEach((st, idx) => {
    const group = new THREE.Group();

    // Glass Slab Geometry
    const slabGeo = new THREE.BoxGeometry(1.65, 2.35, 0.28);
    const slabMat = new THREE.MeshPhysicalMaterial({
      color: st.color,
      metalness: 0.15,
      roughness: 0.1,
      transmission: 0.85,
      thickness: 1.2,
      transparent: true,
      opacity: 0.82,
      reflectivity: 0.9,
      clearcoat: 1.0,
      clearcoatRoughness: 0.1
    });

    const mesh = new THREE.Mesh(slabGeo, slabMat);
    group.add(mesh);

    // Glowing Neon Edge Rim
    const wireGeo = new THREE.EdgesGeometry(slabGeo);
    const wireMat = new THREE.LineBasicMaterial({
      color: st.color,
      linewidth: 2,
      transparent: true,
      opacity: 0.85
    });
    const wireframe = new THREE.LineSegments(wireGeo, wireMat);
    group.add(wireframe);

    // Internal decorative floating elements
    if (idx === 1) { // Tokens
      for (let k = 0; k < 4; k++) {
        const miniCube = new THREE.Mesh(
          new THREE.BoxGeometry(0.28, 0.28, 0.28),
          new THREE.MeshStandardMaterial({ color: 0x38bdf8, emissive: 0x0284c7, roughness: 0.3 })
        );
        miniCube.position.set((k % 2 === 0 ? -0.3 : 0.3), (k < 2 ? 0.3 : -0.3), 0);
        group.add(miniCube);
      }
    } else if (idx === 2) { // Embeddings Matrix Dots
      const dotGroup = new THREE.Group();
      for (let r = 0; r < 5; r++) {
        for (let c = 0; c < 4; c++) {
          const sphere = new THREE.Mesh(
            new THREE.SphereGeometry(0.045, 8, 8),
            new THREE.MeshBasicMaterial({ color: 0xa855f7 })
          );
          sphere.position.set((c - 1.5) * 0.25, (r - 2) * 0.25, 0.05);
          dotGroup.add(sphere);
        }
      }
      group.add(dotGroup);
    } else if (idx === 3) { // Transformer Layers
      for (let l = 0; l < 3; l++) {
        const layerPlate = new THREE.Mesh(
          new THREE.BoxGeometry(1.2, 0.15, 0.6),
          new THREE.MeshStandardMaterial({ color: 0xf472b6, emissive: 0xdb2777, roughness: 0.2 })
        );
        layerPlate.position.set(0, (l - 1) * 0.45, 0);
        group.add(layerPlate);
      }
    }

    group.position.set(startX + idx * spacing, 0, 0);
    group.userData = { id: idx, info: st, baseY: 0 };
    cardGroup.add(group);
    stageCards.push(group);
  });

  scene.add(cardGroup);

  // Connecting Light Energy Particles
  const particleCount = 200;
  const particleGeo = new THREE.BufferGeometry();
  const positions = new Float32Array(particleCount * 3);
  const colors = new Float32Array(particleCount * 3);

  for (let i = 0; i < particleCount; i++) {
    const t = (i / particleCount) * (stageData.length - 1) * spacing + startX;
    positions[i * 3] = t + (Math.random() - 0.5) * 0.4;
    positions[i * 3 + 1] = (Math.random() - 0.5) * 0.6;
    positions[i * 3 + 2] = (Math.random() - 0.5) * 0.8;

    colors[i * 3] = 0.2 + (i / particleCount) * 0.8;
    colors[i * 3 + 1] = 0.7;
    colors[i * 3 + 2] = 1.0;
  }

  particleGeo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  particleGeo.setAttribute("color", new THREE.BufferAttribute(colors, 3));

  const particleMat = new THREE.PointsMaterial({
    size: 0.06,
    vertexColors: true,
    transparent: true,
    opacity: 0.8,
    blending: THREE.AdditiveBlending
  });

  energyParticles = new THREE.Points(particleGeo, particleMat);
  scene.add(energyParticles);

  // Interactive mouse orbit / tilt
  let mouseX = 0, mouseY = 0;
  window.addEventListener("mousemove", (e) => {
    const rect = container.getBoundingClientRect();
    if (e.clientY >= rect.top && e.clientY <= rect.bottom) {
      mouseX = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      mouseY = -(((e.clientY - rect.top) / rect.height) * 2 - 1);
    }
  });

  // Click on stage blocks
  const raycaster = new THREE.Raycaster();
  const mouseVec = new THREE.Vector2();
  container.addEventListener("click", (e) => {
    const rect = container.getBoundingClientRect();
    mouseVec.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    mouseVec.y = -(((e.clientY - rect.top) / rect.height) * 2 - 1);
    raycaster.setFromCamera(mouseVec, camera);

    const intersects = raycaster.intersectObjects(cardGroup.children, true);
    if (intersects.length > 0) {
      let rootCard = intersects[0].object;
      while (rootCard.parent && rootCard.parent !== cardGroup) {
        rootCard = rootCard.parent;
      }
      if (rootCard.userData && typeof rootCard.userData.id === "number") {
        selectStage(rootCard.userData.id);
      }
    }
  });

  // Animation Loop
  let clock = new THREE.Clock();
  function animate() {
    requestAnimationFrame(animate);
    const delta = clock.getDelta();
    const time = clock.getElapsedTime();

    // Gentle camera parallax
    camera.position.x += (mouseX * 1.5 - camera.position.x) * 0.05;
    camera.position.y += (mouseY * 0.8 + 1.2 - camera.position.y) * 0.05;
    camera.lookAt(0, 0, 0);

    // Floating animation on stage glass cards
    stageCards.forEach((c, idx) => {
      c.position.y = Math.sin(time * 2 + idx * 0.8) * 0.12;
      c.rotation.y = THREE.MathUtils.degToRad(-12 + idx * 6) + Math.sin(time + idx) * 0.04;
      c.rotation.x = Math.cos(time * 1.5 + idx) * 0.03;
    });

    // Flowing energy beam
    if (energyParticles) {
      const pos = energyParticles.geometry.attributes.position.array;
      for (let i = 0; i < particleCount; i++) {
        pos[i * 3 + 1] += Math.sin(time * 4 + i) * 0.003;
        pos[i * 3] += 0.015;
        if (pos[i * 3] > -startX) {
          pos[i * 3] = startX;
        }
      }
      energyParticles.geometry.attributes.position.needsUpdate = true;
    }

    renderer.render(scene, camera);
  }

  animate();
  isThreeInitialized = true;

  window.addEventListener("resize", () => {
    if (!container || !renderer || !camera) return;
    const w = container.clientWidth;
    const h = container.clientHeight;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
  });
}

function selectStage(index) {
  state.activeStageStep = index;
  // Update indicator dots
  document.querySelectorAll(".stage-step").forEach((el, idx) => {
    el.classList.toggle("active", idx === index);
  });

  // Pulse corresponding stage card in Three.js
  if (stageCards[index]) {
    const target = stageCards[index];
    const origScale = 1;
    target.scale.set(1.15, 1.15, 1.15);
    setTimeout(() => {
      target.scale.set(origScale, origScale, origScale);
    }, 350);
  }

  // Scroll smoothly to section if applicable
  const sectionIds = ["stage-hero", "stage-tokenizer", "stage-embeddings", "stage-attention", "stage-decoder"];
  const targetId = sectionIds[index];
  const el = document.getElementById(targetId);
  if (el) el.scrollIntoView({ behavior: "smooth" });
}

// =====================================================================
// PART 2: TOKENIZER & BPE VOCABULARY ENGINE
// =====================================================================

function splitIntoWords(text) {
  let t = text.toLowerCase();
  t = t.replace(/([.,!?'])/g, " $1 ");
  return t.split(/\s+/).filter(w => w !== "");
}

function buildVocabulary() {
  const words = splitIntoWords(document.getElementById("corpus").value);

  state.wordCounts = {};
  for (const w of words) {
    state.wordCounts[w] = (state.wordCounts[w] || 0) + 1;
  }
  state.sortedWords = Object.keys(state.wordCounts).sort((a, b) => state.wordCounts[b] - state.wordCounts[a]);

  const letters = [...new Set(words.join("").replace(/[^a-z]/g, ""))].sort();
  state.subwordPieces = [...state.COMMON_ENDINGS, ...letters, ...letters.map(ch => "##" + ch)];

  const slider = document.getElementById("vocab-slider");
  slider.max = state.sortedWords.length;
  slider.value = state.sortedWords.length;
  document.getElementById("vocab-slider-box").style.display = "flex";

  applyVocabSize(state.sortedWords.length);
}

function applyVocabSize(n) {
  const kept = state.sortedWords.slice(0, n);
  document.getElementById("vocab-slider-value").textContent = `${n} of ${state.sortedWords.length} words`;

  state.vocab = { "[UNK]": 0 };
  let rows = "<tr><td>0</td><td>[UNK]</td><td>-</td><td>Special out-of-vocab</td></tr>";
  kept.forEach(w => {
    state.vocab[w] = Object.keys(state.vocab).length;
    rows += `<tr><td>${state.vocab[w]}</td><td><strong style="color:var(--cyan-accent);">${escapeHtml(w)}</strong></td><td>${state.wordCounts[w]}</td><td>Whole Word</td></tr>`;
  });
  state.subwordPieces.forEach(p => {
    if (p in state.vocab) return;
    state.vocab[p] = Object.keys(state.vocab).length;
    rows += `<tr><td>${state.vocab[p]}</td><td>${escapeHtml(p)}</td><td>subword</td><td>Morpheme piece</td></tr>`;
  });

  document.getElementById("vocab-output").innerHTML =
    `<p class="stats">
       <span>Active Vocabulary Size: <strong>${Object.keys(state.vocab).length} tokens</strong></span>
       <span>Kept Words: ${kept.length}</span>
       <span>Subword Fallbacks: ${state.subwordPieces.length}</span>
     </p>
     <div class="table-box"><table>
       <tr><th>ID</th><th>Token</th><th>Corpus Frequency</th><th>Token Type</th></tr>${rows}
     </table></div>`;
}

function splitIntoSubwords(word) {
  const pieces = [];
  let start = 0;
  while (start < word.length) {
    let found = null;
    for (let end = word.length; end > start; end--) {
      let piece = word.slice(start, end);
      if (start > 0) piece = "##" + piece;
      if (piece in state.vocab) { found = piece; break; }
    }
    if (found === null) return [{ text: "[UNK]", id: 0, unknown: true }];
    pieces.push({ text: found, id: state.vocab[found], unknown: true });
    start += found.replace("##", "").length;
  }
  return pieces;
}

function encodeText(text) {
  const tokens = [];
  for (const word of splitIntoWords(text)) {
    if (word in state.vocab) {
      tokens.push({ text: word, id: state.vocab[word] });
    } else {
      tokens.push(...splitIntoSubwords(word));
    }
  }
  return tokens;
}

function colorForId(id) {
  const hue = (id * 47) % 360;
  return `hsl(${hue}, 75%, 78%)`;
}

function renderChips(tokens) {
  let chips = "";
  for (const t of tokens) {
    chips += `<div class="chip" style="background:${colorForId(t.id)}">
                <div class="text">${escapeHtml(t.text)}</div>
                <div class="id">#${t.id}</div>
              </div>`;
  }
  return `<div class="chips">${chips}</div>`;
}

function tokenizeSentence() {
  if (state.vocab === null) buildVocabulary();
  const text = document.getElementById("encode-input").value;
  const tokens = encodeText(text);
  const chips = renderChips(tokens);

  const hasUnknown = tokens.some(t => t.unknown);
  const charCount = text.length;
  const ratio = tokens.length ? (charCount / tokens.length).toFixed(2) : "0";

  document.getElementById("token-output").innerHTML =
    `${chips}
     ${hasUnknown ? '<div class="note-box" style="margin-top:10px;"><i data-lucide="alert-triangle"></i><span>Unseen words split into longest known subword units (prefixed with <code>##</code>). Tokens retain unique semantic numeric IDs.</span></div>' : ""}
     <div class="stats" style="margin-top:14px;">
       <span>Raw Characters: <strong>${charCount}</strong></span>
       <span>Token Count: <strong>${tokens.length}</strong></span>
       <span>Compression Ratio: <strong>${ratio} chars/token</strong></span>
     </div>`;
  if (window.lucide) lucide.createIcons();
}

function escapeHtml(s) {
  return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

// =====================================================================
// PART 3: EMBEDDINGS & 3D MANIFOLD BENDING
// =====================================================================

function hashString(s) {
  let h = 0;
  for (let i = 0; i < s.length; i++) {
    h = (h * 31 + s.charCodeAt(i)) >>> 0;
  }
  return h;
}

function normalizeVector(v) {
  const len = Math.sqrt(v.reduce((sum, x) => sum + x * x, 0)) || 1;
  return v.map(x => x / len);
}

function spellingVector(sentence) {
  const s = sentence.toLowerCase();
  const v = new Array(64).fill(0);
  for (let i = 0; i + 3 <= s.length; i++) {
    const trigram = s.slice(i, i + 3);
    v[hashString(trigram) % 64] += 1;
  }
  return normalizeVector(v);
}

function dot(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += a[i] * b[i];
  return sum;
}

function similarity(a, b) {
  return dot(a, b) / (Math.sqrt(dot(a, a)) * Math.sqrt(dot(b, b)) || 1);
}

function pcaTo3D(vectors) {
  const n = vectors.length, d = vectors[0].length;
  const mean = new Array(d).fill(0);
  for (const v of vectors) for (let j = 0; j < d; j++) mean[j] += v[j] / n;
  const X = vectors.map(v => v.map((x, j) => x - mean[j]));

  const C = [];
  for (let a = 0; a < d; a++) {
    C.push(new Array(d).fill(0));
    for (let b = 0; b < d; b++) {
      for (let i = 0; i < n; i++) C[a][b] += X[i][a] * X[i][b];
      C[a][b] /= (n - 1);
    }
  }

  const components = [];
  for (let k = 0; k < 3; k++) {
    let v = normalizeVector(Array.from({ length: d }, (_, j) => Math.sin(j + 1 + k)));
    for (let iter = 0; iter < 50; iter++) {
      v = normalizeVector(C.map(row => dot(row, v)));
    }
    const lambda = dot(v, C.map(row => dot(row, v)));
    components.push(v);
    for (let a = 0; a < d; a++) {
      for (let b = 0; b < d; b++) C[a][b] -= lambda * v[a] * v[b];
    }
  }

  return X.map(x => components.map(c => dot(x, c)));
}

// Applies 3D manifold warping / bending to the PCA coordinates
function applyManifoldCurvature(points, manifoldType, factor) {
  return points.map(p => {
    let [x, y, z] = p;
    if (manifoldType === "curved") {
      // Hyperbolic saddle-like bending: z' = z + factor * (x^2 - y^2)
      const bend = factor * (x * x - y * y) * 2.5;
      return [x, y, z + bend];
    } else if (manifoldType === "sphere") {
      // Project outward onto spherical manifold
      const r = Math.sqrt(x * x + y * y + z * z) || 1;
      const targetR = 1.2;
      const interp = factor;
      const normX = (x / r) * targetR;
      const normY = (y / r) * targetR;
      const normZ = (z / r) * targetR;
      return [
        x * (1 - interp) + normX * interp,
        y * (1 - interp) + normY * interp,
        z * (1 - interp) + normZ * interp
      ];
    }
    return [x, y, z];
  });
}

function drawPlot(rawPoints, sentences) {
  const curvedPoints = applyManifoldCurvature(rawPoints, state.manifoldType, state.bendFactor);

  // Surface mesh underlay if bending is active
  const traces = [];

  if (state.bendFactor > 0.05) {
    const gridRes = 18;
    const range = 1.5;
    const xGrid = [], yGrid = [], zGrid = [];

    for (let i = 0; i <= gridRes; i++) {
      const rowX = [], rowY = [], rowZ = [];
      const gx = -range + (i / gridRes) * (2 * range);
      for (let j = 0; j <= gridRes; j++) {
        const gy = -range + (j / gridRes) * (2 * range);
        rowX.push(gx);
        rowY.push(gy);
        if (state.manifoldType === "curved") {
          rowZ.push(state.bendFactor * (gx * gx - gy * gy) * 1.5);
        } else if (state.manifoldType === "sphere") {
          const distSq = gx * gx + gy * gy;
          rowZ.push(distSq <= 1.4 ? Math.sqrt(Math.max(0, 1.4 - distSq)) * state.bendFactor : 0);
        } else {
          rowZ.push(0);
        }
      }
      xGrid.push(rowX);
      yGrid.push(rowY);
      zGrid.push(rowZ);
    }

    traces.push({
      type: "surface",
      x: xGrid,
      y: yGrid,
      z: zGrid,
      opacity: 0.28,
      showscale: false,
      colorscale: "Viridis",
      hoverinfo: "skip"
    });
  }

  // Scatter points trace
  const scatterTrace = {
    type: "scatter3d",
    mode: "markers+text",
    x: curvedPoints.map(p => p[0]),
    y: curvedPoints.map(p => p[1]),
    z: curvedPoints.map(p => p[2]),
    text: sentences.map(s => s.length > 20 ? s.slice(0, 20) + "…" : s),
    hovertext: sentences,
    hoverinfo: "text",
    textposition: "top center",
    textfont: {
      family: "Plus Jakarta Sans, sans-serif",
      size: 11,
      color: "#f1f5f9"
    },
    marker: {
      size: sentences.map(s => s === state.userText ? 12 : 7),
      color: sentences.map(s => s === state.userText ? "#f97316" : "#38bdf8"),
      line: {
        color: sentences.map(s => s === state.userText ? "#ffffff" : "rgba(255,255,255,0.7)"),
        width: 2
      },
      opacity: 0.95
    }
  };

  traces.push(scatterTrace);

  const layout = {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    margin: { l: 0, r: 0, t: 0, b: 0 },
    scene: {
      xaxis: {
        title: "PC1 (Latent D1)",
        color: "#94a3b8",
        gridcolor: "rgba(255,255,255,0.06)",
        backgroundcolor: "rgba(0,0,0,0)",
        showbackground: false
      },
      yaxis: {
        title: "PC2 (Latent D2)",
        color: "#94a3b8",
        gridcolor: "rgba(255,255,255,0.06)",
        backgroundcolor: "rgba(0,0,0,0)",
        showbackground: false
      },
      zaxis: {
        title: "PC3 (Curvature D3)",
        color: "#94a3b8",
        gridcolor: "rgba(255,255,255,0.06)",
        backgroundcolor: "rgba(0,0,0,0)",
        showbackground: false
      },
      camera: {
        eye: { x: 1.5, y: 1.5, z: 1.2 }
      }
    }
  };

  const config = { responsive: true, displayModeBar: false };
  Plotly.react("plot", traces, layout, config);
}

function showClosestPairs(vectors, sentences) {
  const pairs = [];
  for (let i = 0; i < vectors.length; i++) {
    for (let j = i + 1; j < vectors.length; j++) {
      pairs.push({ a: sentences[i], b: sentences[j], score: similarity(vectors[i], vectors[j]) });
    }
  }
  pairs.sort((p, q) => q.score - p.score);

  let rows = "";
  for (const p of pairs.slice(0, 5)) {
    const pct = Math.round(Math.max(0, p.score) * 100);
    rows += `<tr>
      <td>${escapeHtml(p.a)}</td>
      <td>${escapeHtml(p.b)}</td>
      <td>
        <span class="badge-value">${p.score.toFixed(3)}</span>
        <div style="width:60px; height:4px; background:rgba(255,255,255,0.1); border-radius:999px; margin-top:4px; overflow:hidden;">
          <div style="width:${pct}%; height:100%; background:linear-gradient(to right, #38bdf8, #8b5cf6);"></div>
        </div>
      </td>
    </tr>`;
  }

  document.getElementById("pairs-output").innerHTML =
    `<div class="card-title-bar" style="margin-top:16px;">
       <div class="title-with-icon">
         <i data-lucide="link-2"></i>
         <span>Top Semantic Nearest Neighbors (Cosine Similarity)</span>
       </div>
     </div>
     <div class="table-box">
       <table><tr><th>Sentence A</th><th>Sentence B</th><th>Cosine Metric</th></tr>${rows}</table>
     </div>`;
  if (window.lucide) lucide.createIcons();
}

let lastComputedVectors = null;
let lastComputedSentences = null;
let rawPcaCoords = null;

async function embedAndPlot() {
  const status = document.getElementById("embed-status");
  status.textContent = "";
  const sentences = document.getElementById("sentences").value
    .split("\n").map(s => s.trim()).filter(s => s !== "");

  if (sentences.length < 4) {
    status.innerHTML = '<span style="color:#ef4444;"><i data-lucide="alert-circle"></i> Please enter at least 4 sentences for 3D PCA projection.</span>';
    if (window.lucide) lucide.createIcons();
    return;
  }

  const vectors = sentences.map(spellingVector);
  lastComputedVectors = vectors;
  lastComputedSentences = sentences;

  rawPcaCoords = pcaTo3D(vectors);
  drawPlot(rawPcaCoords, sentences);
  showClosestPairs(vectors, sentences);

  if (sentences.includes(state.userText)) {
    showUserWordDetails(vectors, sentences);
  }
}

async function addUserWord() {
  const text = document.getElementById("new-word").value.trim();
  if (text === "") return;

  if (state.vocab === null) buildVocabulary();
  document.getElementById("new-word-output").innerHTML =
    `<p class="stats" style="margin-top:8px;">Subword Tokenization of Probe:</p>${renderChips(encodeText(text))}
     <div id="new-word-vector"></div>`;

  const box = document.getElementById("sentences");
  const lines = box.value.split("\n").map(s => s.trim());
  if (!lines.includes(text)) box.value = box.value.trimEnd() + "\n" + text;
  state.userText = text;
  await embedAndPlot();
}

function showUserWordDetails(vectors, sentences) {
  const me = sentences.indexOf(state.userText);
  if (me === -1) return;
  const v = vectors[me];

  const others = sentences
    .map((s, i) => ({ s, score: similarity(v, vectors[i]) }))
    .filter((o, i) => i !== me)
    .sort((a, b) => b.score - a.score)
    .slice(0, 3);

  let rows = "";
  for (const o of others) {
    rows += `<tr><td>${escapeHtml(o.s)}</td><td><span class="badge-value">${o.score.toFixed(3)}</span></td></tr>`;
  }

  const container = document.getElementById("new-word-vector");
  if (!container) return;
  container.innerHTML =
    `<p class="stats" style="margin-top:8px;">Embedding Sample (first 8 dims of 64-D):</p>
     <div style="background:rgba(0,0,0,0.4); padding:8px 12px; border-radius:8px; font-family:var(--font-mono); font-size:0.75rem; color:#38bdf8;">
       [${v.slice(0, 8).map(x => x.toFixed(3)).join(", ")}, …]
     </div>
     <p class="stats" style="margin-top:8px;">Nearest neighbors to probe:</p>
     <div class="table-box"><table><tr><th>Sentence</th><th>Similarity</th></tr>${rows}</table></div>`;
}

// =====================================================================
// PART 4: MULTI-HEAD ATTENTION HEATMAP
// =====================================================================

const headDescriptions = [
  "<strong>Head 1: Syntactic Dependencies:</strong> Focuses on neighboring verbs, prepositions, and structural grammar ties.",
  "<strong>Head 2: Coreference Resolution:</strong> Connects pronouns like 'it' or 'they' back to their true antecedent entity ('animal' vs 'street').",
  "<strong>Head 3: Positional Locality:</strong> Attends strongly to immediate predecessor tokens, simulating causal local n-gram context.",
  "<strong>Head 4: Semantic Context & Predicates:</strong> Couples action predicates ('tired', 'slept') with agent nouns."
];

function computeAttentionWeights(words, headIndex) {
  const n = words.length;
  const matrix = [];

  for (let i = 0; i < n; i++) {
    const row = [];
    const fromWord = words[i].toLowerCase();

    for (let j = 0; j < n; j++) {
      const toWord = words[j].toLowerCase();
      let weight = 0.05; // baseline

      if (headIndex === 0) { // Syntactic
        if (Math.abs(i - j) === 1) weight += 0.45;
        if (fromWord === "cross" && toWord === "street") weight += 0.6;
        if (fromWord === "tired" && toWord === "was") weight += 0.5;
      } else if (headIndex === 1) { // Coreference
        if (fromWord === "it" && toWord === "animal") weight += 0.82;
        else if (fromWord === "it" && toWord === "street") weight += 0.12;
        else if (fromWord === "tired" && toWord === "animal") weight += 0.65;
        else if (i === j) weight += 0.3;
      } else if (headIndex === 2) { // Positional
        const dist = i - j;
        if (dist >= 0 && dist <= 2) weight += 0.75 / (dist + 1);
      } else { // Semantic
        if ((fromWord === "animal" || fromWord === "tired") && (toWord === "animal" || toWord === "tired")) weight += 0.7;
        if ((fromWord === "street" || fromWord === "cross") && (toWord === "street" || toWord === "cross")) weight += 0.65;
        if (i === j) weight += 0.25;
      }

      // Add pseudo-random deterministic jitter
      const jitter = (Math.sin(i * 13 + j * 17) * 0.5 + 0.5) * 0.1;
      row.push(weight + jitter);
    }

    // Softmax normalization across row
    const exps = row.map(w => Math.exp(w * 2.5));
    const sumExp = exps.reduce((a, b) => a + b, 0);
    matrix.push(exps.map(e => e / sumExp));
  }

  return matrix;
}

function renderAttentionHeatmap() {
  const text = document.getElementById("attention-sentence-input").value;
  const words = splitIntoWords(text);
  if (words.length === 0) return;

  const weights = computeAttentionWeights(words, state.activeAttentionHead);

  // Render HTML Matrix
  let tableHtml = '<table class="attention-matrix-table">';
  // Header row
  tableHtml += '<tr><th></th>' + words.map(w => `<th>${escapeHtml(w)}</th>`).join('') + '</tr>';

  for (let i = 0; i < words.length; i++) {
    tableHtml += `<tr><th>${escapeHtml(words[i])}</th>`;
    for (let j = 0; j < words.length; j++) {
      const val = weights[i][j];
      const opacity = Math.min(1, Math.max(0.08, val));
      const bg = `rgba(236, 72, 153, ${opacity})`;
      const textCol = opacity > 0.4 ? '#ffffff' : 'rgba(255,255,255,0.4)';
      tableHtml += `<td class="attention-cell" style="background:${bg}; color:${textCol};" title="${words[i]} &rarr; ${words[j]}: ${val.toFixed(3)}">${val.toFixed(2)}</td>`;
    }
    tableHtml += '</tr>';
  }
  tableHtml += '</table>';

  document.getElementById("attention-heatmap-container").innerHTML = tableHtml;

  // Render top dependency linkages for key tokens
  const linksContainer = document.getElementById("token-dependency-visualizer");
  let linksHtml = '';
  // Highlight top 4 strongest cross-word attention pairs (i != j)
  const allPairs = [];
  for (let i = 0; i < words.length; i++) {
    for (let j = 0; j < words.length; j++) {
      if (i !== j) {
        allPairs.push({ from: words[i], to: words[j], weight: weights[i][j] });
      }
    }
  }
  allPairs.sort((a, b) => b.weight - a.weight);

  for (const pair of allPairs.slice(0, 5)) {
    const pct = Math.round(pair.weight * 100);
    linksHtml += `<div class="token-link-row">
      <span><strong>${escapeHtml(pair.from)}</strong> &rarr; <span style="color:var(--cyan-accent);">${escapeHtml(pair.to)}</span></span>
      <div style="display:flex; align-items:center; gap:8px;">
        <span class="badge-value">${pair.weight.toFixed(3)}</span>
        <div class="token-link-weight-bar">
          <div class="token-link-fill" style="width:${pct}%;"></div>
        </div>
      </div>
    </div>`;
  }
  linksContainer.innerHTML = linksHtml;

  // Update explanation
  document.getElementById("head-desc-text").innerHTML = headDescriptions[state.activeAttentionHead];
}

// =====================================================================
// PART 5: DECODER & AUTOREGRESSIVE GENERATION
// =====================================================================

const candidateVocabulary = [
  { word: "jumps", baseLogit: 4.8 },
  { word: "leaps", baseLogit: 4.2 },
  { word: "runs", baseLogit: 3.5 },
  { word: "sleeps", baseLogit: 2.1 },
  { word: "howls", baseLogit: 1.8 },
  { word: "barks", baseLogit: 1.2 },
  { word: "over", baseLogit: 3.9 },
  { word: "silently", baseLogit: 2.6 }
];

function computeNextTokenProbabilities() {
  const temp = state.temperature;
  const topK = state.topK;

  // Apply temperature scaling: z_i / T
  const scaled = candidateVocabulary.map(item => ({
    word: item.word,
    scaledLogit: item.baseLogit / temp
  }));

  // Sort descending by logit
  scaled.sort((a, b) => b.scaledLogit - a.scaledLogit);

  // Top-K Filtering
  const pool = scaled.slice(0, topK);

  // Softmax
  const maxLogit = pool[0].scaledLogit;
  const exps = pool.map(item => Math.exp(item.scaledLogit - maxLogit));
  const sumExps = exps.reduce((a, b) => a + b, 0);

  return pool.map((item, idx) => ({
    word: item.word,
    prob: exps[idx] / sumExps
  }));
}

function renderProbabilities() {
  const probs = computeNextTokenProbabilities();
  let barsHtml = "";

  probs.forEach((p, idx) => {
    const pct = (p.prob * 100).toFixed(1);
    const isTop = idx === 0;
    const barGradient = isTop
      ? "linear-gradient(to right, #38bdf8, #8b5cf6)"
      : "linear-gradient(to right, rgba(255,255,255,0.2), rgba(255,255,255,0.4))";

    barsHtml += `
      <div class="prob-item">
        <div class="prob-meta">
          <span>${isTop ? '★ ' : ''}<strong>${escapeHtml(p.word)}</strong></span>
          <span style="color:${isTop ? 'var(--cyan-accent)' : 'var(--text-muted)'};">${pct}%</span>
        </div>
        <div class="prob-bar-track">
          <div class="prob-bar-fill" style="width:${pct}%; background:${barGradient};"></div>
        </div>
      </div>
    `;
  });

  document.getElementById("probabilities-bars").innerHTML = barsHtml;
}

function stepGenerateToken() {
  const probs = computeNextTokenProbabilities();
  // Sample token according to distribution
  const r = Math.random();
  let cumulative = 0;
  let chosenWord = probs[0].word;

  for (const item of probs) {
    cumulative += item.prob;
    if (r <= cumulative) {
      chosenWord = item.word;
      break;
    }
  }

  const stream = document.getElementById("generated-stream");
  const span = document.createElement("span");
  span.className = "new-token";
  span.textContent = " " + chosenWord;
  stream.appendChild(span);

  // Slightly perturb base logits for next autoregressive step
  candidateVocabulary.forEach(c => {
    c.baseLogit += (Math.random() - 0.48) * 1.5;
  });

  renderProbabilities();
}

// =====================================================================
// INITIALIZATION & EVENT LISTENERS
// =====================================================================

document.addEventListener("DOMContentLoaded", () => {
  // Initialize Three.js 3D Pipeline
  initThreePipeline();

  // Initialize Vocabulary and Tokenizer
  buildVocabulary();
  tokenizeSentence();

  // Initialize Embeddings & 3D Plot
  embedAndPlot();

  // Initialize Attention Mechanism
  renderAttentionHeatmap();

  // Initialize Decoder
  renderProbabilities();

  // Navigation tab clicks
  document.querySelectorAll(".nav-tab").forEach(tab => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".nav-tab").forEach(t => t.classList.remove("active"));
      tab.classList.add("active");
      const targetId = tab.dataset.target;
      const el = document.getElementById(targetId);
      if (el) el.scrollIntoView({ behavior: "smooth" });
    });
  });

  // Showcase interactive indicators
  document.querySelectorAll(".stage-step").forEach(step => {
    step.addEventListener("click", () => {
      const idx = parseInt(step.dataset.step, 10);
      selectStage(idx);
    });
  });

  // Toggle View in 3D header
  document.getElementById("view-3d-btn").addEventListener("click", function() {
    this.classList.add("active");
    document.getElementById("view-flow-btn").classList.remove("active");
    if (camera) camera.position.set(0, 1.5, 9.5);
  });

  document.getElementById("view-flow-btn").addEventListener("click", function() {
    this.classList.add("active");
    document.getElementById("view-3d-btn").classList.remove("active");
    if (camera) camera.position.set(0, 6, 8);
  });

  // Quick prompt run from showcase
  document.getElementById("run-pipeline-quick-btn").addEventListener("click", () => {
    const val = document.getElementById("pipeline-quick-input").value;
    document.getElementById("encode-input").value = val;
    tokenizeSentence();
    document.getElementById("new-word").value = val;
    addUserWord();
    document.getElementById("attention-sentence-input").value = val;
    renderAttentionHeatmap();
    selectStage(1);
  });

  // Simulate full pipeline
  document.getElementById("play-full-pipeline-btn").addEventListener("click", () => {
    let curr = 0;
    const interval = setInterval(() => {
      selectStage(curr);
      curr++;
      if (curr >= 5) clearInterval(interval);
    }, 1200);
  });

  // Tokenizer Listeners
  document.getElementById("build-vocab-btn").addEventListener("click", buildVocabulary);
  document.getElementById("tokenize-btn").addEventListener("click", tokenizeSentence);
  document.getElementById("vocab-slider").addEventListener("input", function() {
    applyVocabSize(Number(this.value));
    tokenizeSentence();
  });

  document.querySelectorAll(".preset-chip").forEach(chip => {
    chip.addEventListener("click", () => {
      document.getElementById("encode-input").value = chip.dataset.text;
      tokenizeSentence();
    });
  });

  document.getElementById("reset-corpus-btn").addEventListener("click", () => {
    document.getElementById("corpus").value = `The cat sat on the mat.\nThe dog sat on the rug.\nThe cat saw the dog!\nA dog and a cat can be friends.\nDo you like the cat?\nI don't like the rain, but I like the sun.`;
    buildVocabulary();
    tokenizeSentence();
  });

  // Embedding Listeners
  document.getElementById("embed-btn").addEventListener("click", embedAndPlot);
  document.getElementById("add-word-btn").addEventListener("click", addUserWord);
  document.getElementById("new-word").addEventListener("keydown", e => { if (e.key === "Enter") addUserWord(); });

  // Manifold Bending & Projection Controls
  document.querySelectorAll(".manifold-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".manifold-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      state.manifoldType = btn.dataset.manifold;
      if (rawPcaCoords && lastComputedSentences) {
        drawPlot(rawPcaCoords, lastComputedSentences);
      }
    });
  });

  document.getElementById("bend-slider").addEventListener("input", function() {
    state.bendFactor = parseFloat(this.value);
    document.getElementById("bend-factor-label").textContent = state.bendFactor.toFixed(2);
    if (rawPcaCoords && lastComputedSentences) {
      drawPlot(rawPcaCoords, lastComputedSentences);
    }
  });

  // Attention Listeners
  document.getElementById("compute-attention-btn").addEventListener("click", renderAttentionHeatmap);
  document.querySelectorAll(".head-pill").forEach(pill => {
    pill.addEventListener("click", () => {
      document.querySelectorAll(".head-pill").forEach(p => p.classList.remove("active"));
      pill.classList.add("active");
      state.activeAttentionHead = parseInt(pill.dataset.head, 10);
      renderAttentionHeatmap();
    });
  });

  // Decoder Listeners
  document.getElementById("step-generate-btn").addEventListener("click", stepGenerateToken);
  document.getElementById("auto-generate-btn").addEventListener("click", () => {
    let count = 0;
    const genInterval = setInterval(() => {
      stepGenerateToken();
      count++;
      if (count >= 5) clearInterval(genInterval);
    }, 280);
  });

  document.getElementById("temp-slider").addEventListener("input", function() {
    state.temperature = parseFloat(this.value);
    document.getElementById("temp-val").textContent = state.temperature.toFixed(1);
    renderProbabilities();
  });

  document.getElementById("topk-slider").addEventListener("input", function() {
    state.topK = parseInt(this.value, 10);
    document.getElementById("topk-val").textContent = state.topK;
    renderProbabilities();
  });

  document.getElementById("reset-prompt-btn").addEventListener("click", () => {
    const p = document.getElementById("prompt-input").value;
    document.getElementById("generated-stream").textContent = p;
    renderProbabilities();
  });
});
