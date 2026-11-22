# Activity Log & Development History

This file tracks all architectural decisions, implementations, modifications, and updates made to the **LLM Pipeline Explorer** so that any developer or collaborator can quickly inspect the timeline and technical progress.

---

## [2026-11-21] — Initial Redesign & Modern Liquid Glass Architecture

### Author: Antigravity AI (Google DeepMind)
### Goal:
Transform the raw, unstyled prototype into a production-grade interactive educational application inspired by modern **Liquid Glass / Windows 11 Mica / Apple Dark Glass** aesthetics, while expanding the LLM pipeline from just Tokenizer + Embeddings to a full end-to-end visualization (Input → Tokens → Embeddings & 3D Manifold → Multi-Head Attention → Decoder Logits).

### Key Accomplishments & Files Created:

1. **`index.html`**:
   - Structured the page using a dark cyber-glass shell (`.app-viewport`, `.ambient-glow`).
   - Integrated **Three.js (r128)** for 3D stage showcase rendering, **Plotly.js (v2.35.2)** for 3D scatter & manifold plotting, and **Lucide Icons**.
   - Added Top Navigation Bar with glass pill tabs, status lights, and interactive quick action buttons.
   - Built the 3D Pipeline Hero Showcase with interactive live stage indicator pills.
   - Added 4 feature cards matching the user's reference mockup.
   - Structured 4 dedicated functional pipeline labs:
     - **Stage 01:** Subword Tokenizer & BPE Lab
     - **Stage 02:** 3D Embeddings, Semantic Distance & 3D Manifold Curvature
     - **Stage 03:** Multi-Head Self-Attention ($Q, K, V$ Softmax matrix)
     - **Stage 04:** Autoregressive Next-Token Decoder & Temperature / Top-K sampling

2. **`style.css`**:
   - Implemented a complete CSS variable system (`--bg-core`, `--bg-surface`, `--border-glass`, `--cyan-accent`, `--purple-neon`, `--magenta-neon`, `--amber-neon`).
   - Designed 3 floating ambient radial blur glow orbs (`.ambient-glow`) with subtle keyframe breathing animations.
   - Applied frosted glassmorphism backdrop filters (`backdrop-filter: blur(24px);`).
   - Styled glass cards, sliders, token chips with dynamic HSL color mapping, responsive data tables, and custom scrollbars.
   - Built responsive layouts for desktop, tablet, and mobile displays.

3. **`script.js`**:
   - **WebGL Three.js Canvas**:
     - Built 5 physical glass slabs (`MeshPhysicalMaterial`) with light transmission, roughness, neon edge outlines (`EdgesGeometry`), and internal stage geometry (tokens cubes, embedding grid dots, transformer layer plates).
     - Added mouse parallax, hovering sinusoidal motion, and raycasting for clicking 3D stage blocks.
     - Added an energy beam with 200 flowing particle vectors.
   - **Tokenizer Engine**:
     - Preserved and enhanced original regex word boundary splitting.
     - Implemented dynamic frequency-sorted vocabulary and subword fallback (`##` affixes and character fragments).
     - Connected live capacity slider with automatic re-tokenization.
   - **Embeddings & 3D Manifold Engine**:
     - Implemented 64-dimensional character trigram hashing with $L_2$ normalization.
     - Implemented power-iteration PCA with matrix deflation down to 3 principal components (PC1, PC2, PC3).
     - **New Feature: 3D Manifold Warping**: Added hyperbolic saddle curvature ($z' = z + f \cdot (x^2 - y^2)$) and hypersphere projection with interactive 3D surface mesh underlays.
   - **Multi-Head Self-Attention Lab**:
     - Modeled 4 attention heads (Syntactic, Coreference resolution for pronouns like *"it"*, Positional locality, and Semantic predicates).
     - Softmax attention weight computation and live heatmap matrix rendering with tooltips.
     - Extracted top cross-token dependency linkages with gradient strength bars.
   - **Decoder & Sampling Engine**:
     - Simulated autoregressive next-token logits.
     - Applied temperature scaling ($z_i / T$) and Top-K candidate pool filtering.
     - Step-by-step token appending with animation glow.

---

## Log Template for Future Entries
```markdown
## [YYYY-MM-DD] — <Feature / Fix Title>
### Developer: <Name or Role>
### Changes Made:
- File modified: Description of change
- Architecture rationale: Why it was implemented this way
- Verification: Tests run or visual check performed
```
