# Engineering Roadmap & Developer Handover Notes

> **Purpose:** This file serves as the architectural blueprint and developer handover documentation. If another engineer takes over or joins this project, they can read this file to understand the system design, current limitations, planned features, and implementation roadmap.

---

## 1. System Architecture Overview

```
                      ┌──────────────────────────────────────┐
                      │          LLM Pipeline UI             │
                      │  (index.html + style.css: Glass UI)  │
                      └──────────────────┬───────────────────┘
                                         │
        ┌────────────────────────────────┼────────────────────────────────┐
        │                                │                                │
┌───────▼────────┐              ┌────────▼────────┐              ┌────────▼────────┐
│  Three.js 3D   │              │  Plotly.js 3D   │              │  Vanilla JS Core│
│ Glass Pipeline │              │ Manifold & PCA  │              │ Tokenizer, Attn,│
│  Stage Visual  │              │ Latent Clusters │              │ Logits Decoder  │
└────────────────┘              └─────────────────┘              └─────────────────┘
```

The application is built deliberately with zero heavy build steps (pure HTML5, CSS3, Vanilla ES6+ JavaScript, Three.js, and Plotly) to guarantee:
1. Instant client-side execution in any browser without needing `npm install` or node bundlers.
2. High portability and embeddability.
3. Smooth 60fps animations with hardware-accelerated WebGL.

---

## 2. Component Structure & Data Flow

### A. Tokenizer Stage (`script.js` -> `buildVocabulary`, `encodeText`)
- **Input:** Raw text corpus and single sentence to encode.
- **Process:**
  1. Regex punctuation padding and word segmentation.
  2. Word frequency tallying and BPE vocabulary truncation via slider.
  3. Longest-match greedy subword search with WordPiece-style `##` continuation markers.
- **Output:** Color-mapped token chips and character-to-token compression ratio.

### B. Embeddings & 3D Manifold Stage (`script.js` -> `spellingVector`, `pcaTo3D`, `applyManifoldCurvature`)
- **Input:** Multiple text sentences.
- **Process:**
  1. Converts sentences into 64-dimensional character trigram hash counts.
  2. Centers and normalizes into an $L_2$ unit hypersphere.
  3. Computes the covariance matrix $C$ ($64 \times 64$) and extracts the top 3 eigenvectors using 50-iteration power iteration with Hotelling deflation.
  4. Applies non-Euclidean manifold warping:
     - **Flat:** Standard Euclidean PCA projection.
     - **Hyperbolic:** Warps $z$-axis with $f \cdot (x^2 - y^2)$ simulating saddle geometry.
     - **Hypersphere:** Normalizes onto a curved sphere envelope.
- **Output:** Interactive 3D Plotly scatter with wireframe surface mesh and cosine nearest neighbors.

### C. Self-Attention Stage (`script.js` -> `computeAttentionWeights`, `renderAttentionHeatmap`)
- **Input:** Natural language sentence with ambiguous pronouns or predicates.
- **Process:** Simulates 4 specialized transformer attention heads ($Q, K, V$ scaled dot-product attention followed by Softmax).
- **Output:** Dynamic heat matrix and token-to-token dependency bar charts.

### D. Decoder & Sampling Stage (`script.js` -> `computeNextTokenProbabilities`, `stepGenerateToken`)
- **Input:** Prompt stream, Temperature $T \in [0.1, 1.5]$, Top-K $\in [1, 8]$.
- **Process:** Logit scaling $z_i / T$, softmax distribution, and cumulative distribution sampling.
- **Output:** Autoregressively generated text stream and live animated probability bars.

---

## 3. Planned Features & Next Development Phases

### Phase 1: Real-time Transformer Layer Stepper (Upcoming)
- [ ] Add a **Step-by-Step Transformer Walkthrough Modal**:
  - Show how a token vector is updated through Layer 1, Layer 2, Layer 3 with **Residual Connections** ($x + \text{Attention}(x)$) and **Layer Normalization** ($\text{LayerNorm}$).
  - Visually depict the residual stream passing through Feed-Forward Network (FFN) expansion.

### Phase 2: WebGPU / Transformers.js Integration
- [ ] Connect optional **Transformers.js** (Hugging Face ONNX in browser):
  - Allow users to switch between the lightweight educational simulation mode and a real micro-model (e.g. `Xenova/gpt2` or `all-MiniLM-L6-v2` running 100% locally in browser WebAssembly/WebGPU).
  - Extract true contextual token embeddings rather than n-gram approximations when enabled.

### Phase 3: Positional Encoding Visualizer (RoPE & Sinusoidal)
- [ ] Add an interactive sub-tab explaining **Positional Encoding**:
  - Interactive heatmaps of $\sin(\text{pos} / 10000^{2i/d})$ and $\cos(\text{pos} / 10000^{2i/d})$.
  - Visual explanation of Rotary Position Embeddings (RoPE) showing 2D rotation of key-query vectors.

### Phase 4: UI Enhancements & Audio/Visual Feedback
- [ ] Add subtle sound synthesizer effects (using Web Audio API) on token emission and button clicks.
- [ ] Add interactive camera orbit presets for the 3D pipeline (Isometric, Top-Down, First-Person Token View).

---

## 4. Key Notes for the Next Developer

1. **Working Directory:**
   The workspace path is `/Users/aniket/llm Pipline ` (note the space in the folder name). Always use quotes when running terminal commands.
2. **Third-Party Libraries:**
   All dependencies are loaded via stable CDNs in `index.html`:
   - `Three.js r128`
   - `Plotly.js 2.35.2`
   - `Lucide Icons`
   - `Google Fonts (Plus Jakarta Sans & JetBrains Mono)`
3. **State Management:**
   All shared variables and configuration parameters reside in the centralized `state` object at the top of `script.js`. If you add new parameters, add them to `state` first.
4. **Style Consistency:**
   Whenever creating new cards or modals, make sure to reuse the CSS classes `.glass-card`, `.card-inner`, and the CSS variables `--bg-card`, `--border-glass`, and `--cyan-accent` to preserve the liquid glass aesthetic.
