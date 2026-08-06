# DSCA-ViT — Refined Architecture Blueprint (v2)
## Dual-Stain Cross-Attention Vision Transformer for HER2 IHC Scoring

> **Phase 1 — Final Architecture Design**
> Incorporates all feedback from initial review. Ready for implementation.

---

## 1. Design Philosophy

Every architectural decision is driven by a single principle:

> **HER2 IHC images are not natural images.
> The architecture must encode what pathologists already know.**

Three biological facts drive the design:

| Fact | Architectural Consequence |
|:---|:---|
| H and DAB encode *different* biological signals | → Dual-stream processing |
| Both stains originate from the *same* tissue section | → Shared encoder (same spatial features) |
| Pixel $(x,y)$ in H corresponds *exactly* to pixel $(x,y)$ in DAB | → **Spatially-biased cross-attention** |

The third fact — **perfect spatial registration** — is the core novelty that most multimodal architectures cannot exploit because their modalities are not pixel-aligned.

---

## 2. Final Architecture

```
                        ┌─────────────────────┐
                        │   RGB Patch (224²×3) │
                        └──────────┬──────────┘
                                   │
                        ┌──────────▼──────────┐
                        │  Color Deconvolution │
                        │  (Ruifrok, fixed)    │
                        └──────────┬──────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │                             │
          ┌─────────▼─────────┐         ┌─────────▼─────────┐
          │   H Channel       │         │   DAB Channel      │
          │   (224²×1)        │         │   (224²×1)         │
          └─────────┬─────────┘         └─────────┬─────────┘
                    │                             │
          ┌─────────▼─────────┐         ┌─────────▼─────────┐
          │  Proj_H: 1→3 ch   │         │  Proj_D: 1→3 ch   │
          │  Conv2d(1,3,1×1)  │         │  Conv2d(1,3,1×1)  │
          └─────────┬─────────┘         └─────────┬─────────┘
                    │                             │
                    │   ┌─────────────────────┐   │
                    ├──►│  Shared ViT-B/16    │◄──┤
                    │   │  Blocks 1–9         │   │
                    │   │  (100% shared)      │   │
                    │   └──────────┬──────────┘   │
                    │              │               │
          ┌─────────▼────┐                 ┌──────▼────────┐
          │  H₉ Tokens   │                 │  D₉ Tokens    │
          │  (197×768)   │                 │  (197×768)    │
          └──────┬───────┘                 └──────┬────────┘
                 │                                │
          ┌──────▼────────────────────────────────▼────────┐
          │    SPATIALLY-BIASED BIDIRECTIONAL               │
          │    CROSS-ATTENTION MODULE                       │
          │                                                │
          │    (attention biased toward corresponding       │
          │     spatial positions)                          │
          └──────┬────────────────────────────────┬────────┘
                 │                                │
          ┌──────▼───────┐                 ┌──────▼────────┐
          │  Ĥ Tokens    │                 │  D̂ Tokens     │
          │  (197×768)   │                 │  (197×768)    │
          └──────┬───────┘                 └──────┬────────┘
                 │                                │
                 │   ┌─────────────────────┐      │
                 ├──►│  Shared ViT-B/16    │◄─────┤
                 │   │  Blocks 10–12       │      │
                 │   │  (100% shared)      │      │
                 │   └──────────┬──────────┘      │
                 │              │                  │
          ┌──────▼───────┐                 ┌──────▼────────┐
          │  H_final     │                 │  D_final      │
          │  (197×768)   │                 │  (197×768)    │
          └──────┬───────┘                 └──────┬────────┘
                 │                                │
          ┌──────▼────────────────────────────────▼────────┐
          │           TOKEN FUSION                         │
          │           (Gated per-token)                    │
          └────────────────────┬───────────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Fused Tokens       │
                    │  (197×768)          │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  REFINEMENT BLOCK   │
                    │  (1 Transformer     │
                    │   Block, new init)  │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  CLASSIFICATION     │
                    │  HEAD               │
                    └──────────┬──────────┘
                               │
                          HER2 Score
                        {0, 1+, 2+, 3+}
```

### What Was Removed (and Why)

| Component | Status | Reason |
|:---|:---|:---|
| RGB residual stream | ❌ Removed | Muddies the contribution — reviewer could attribute gains to RGB |
| Stain-specific adapters | ❌ Removed | Introduces another variable; would be a separate paper |
| Multi-depth cross-attention | ❌ Removed | Single insertion keeps design clean; multi-depth is an ablation |

### What Was Added (and Why)

| Component | Status | Reason |
|:---|:---|:---|
| Spatial correspondence bias | ✅ **Added** | Core novelty — exploits perfect registration |
| Refinement block | ✅ **Added** | Allows reasoning over fused representation |

---

## 3. The Core Novelty — Spatially-Biased Cross-Attention

### 3.1 The Problem with Standard Cross-Attention

Standard multi-head cross-attention computes:

$$\text{Attn}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

This produces a $197 \times 197$ attention matrix where **every H token can attend equally to every DAB token**. The network must *learn* that patch 37 in H should focus on patch 37 in DAB — which is wasteful, because we already *know* this from the physics of the imaging process.

### 3.2 The Insight

In standard multimodal settings (RGB + depth, CT + PET from different scanners), spatial correspondence is approximate at best. But in color deconvolution:

> **Patch $i$ in H is derived from exactly the same $16 \times 16$ pixel region as patch $i$ in DAB.**

This means:
- Patch 37 in H should **strongly** attend to patch 37 in DAB (same tissue location)
- Patch 37 may **weakly** attend to patches 36, 38, 23, 51 (spatial neighbors)
- Patch 37 should **rarely** attend to patch 182 (distant, unrelated tissue)

### 3.3 Three Strategies for Spatial Bias

I propose three strategies, ordered by increasing sophistication. All should be tested in ablation.

---

#### Strategy S1 — Additive Positional Bias (Recommended Starting Point)

Add a **learnable spatial bias matrix** $B \in \mathbb{R}^{197 \times 197}$ to the attention logits:

$$\text{Attn}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}} + B\right) V$$

where $B$ is initialized as:

$$B_{ij} = \begin{cases} +\beta & \text{if } i = j \quad \text{(same spatial position)} \\ -\gamma \cdot d(i,j) & \text{if } i \neq j \quad \text{(distance penalty)} \end{cases}$$

Here $d(i,j)$ is the Euclidean distance between patch $i$ and patch $j$ on the $14 \times 14$ grid, and $\beta, \gamma > 0$ are learnable scalars or hyperparameters.

**Properties**:
- At initialization, attention is strongly biased toward the corresponding patch
- During training, the network can *override* the bias if non-local attention is useful
- $B$ is shared across heads, or can be per-head for richer patterns
- CLS token (index 0) has no spatial position — its bias row/column is initialized to 0 (uniform)

**Parameter cost**: $197 \times 197 = 38{,}809$ parameters (negligible)

---

#### Strategy S2 — Gaussian Spatial Prior (Fixed, Not Learned)

Replace the learnable bias with a fixed Gaussian kernel:

$$B_{ij} = -\frac{d(i,j)^2}{2\sigma^2}$$

where $\sigma$ controls the spatial falloff. This is not learned — it's a pure inductive bias.

**Properties**:
- Stronger inductive bias (cannot be overridden)
- Zero additional parameters
- $\sigma$ is a hyperparameter (try $\sigma \in \{1, 2, 3, 5\}$ in grid units)
- Simpler to justify theoretically

---

#### Strategy S3 — Spatially-Masked Attention (Hardest Constraint)

Restrict each token to attend only to a **local window** of $k \times k$ corresponding patches:

$$\text{Mask}_{ij} = \begin{cases} 0 & \text{if } d(i,j) \leq r \\ -\infty & \text{otherwise} \end{cases}$$

$$\text{Attn}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}} + \text{Mask}\right) V$$

**Properties**:
- Enforces strict locality
- Reduces computation from $O(N^2)$ to $O(N \cdot k^2)$
- Risk: if non-local correspondence matters (e.g., tumor heterogeneity), this hurts

---

#### Strategy Comparison

| Strategy | Params | Computational Cost | Flexibility | Novelty |
|:---|:---|:---|:---|:---|
| **S1 — Learnable bias** | ~39K | Same as standard | High (can override bias) | ⭐⭐⭐⭐ |
| **S2 — Gaussian prior** | 0 | Same as standard | Low (fixed) | ⭐⭐⭐ |
| **S3 — Hard mask** | 0 | Reduced | None (strict cutoff) | ⭐⭐ |
| **S0 — No bias (baseline)** | 0 | Same as standard | Maximum | ⭐ |

> [!IMPORTANT]
> **Recommendation**: Start with **S1 (learnable bias)** as the default architecture. It offers the best balance — the spatial prior is encoded at initialization, but the network retains the freedom to learn non-local patterns if the data supports them. Test S0, S2, S3 as ablations.

### 3.4 Spatial Bias — Initialization Detail

For Strategy S1, the bias matrix is initialized based on the $14 \times 14$ patch grid:

```
Patch layout (14×14 = 196 patches + 1 CLS):

Token 0:  CLS (no spatial position)
Token 1:  grid position (0, 0)
Token 2:  grid position (0, 1)
...
Token 14: grid position (1, 0)
...
Token 196: grid position (13, 13)

Distance d(i,j):
  For tokens i, j ∈ {1,...,196}:
    row_i, col_i = (i-1) // 14, (i-1) % 14
    row_j, col_j = (j-1) // 14, (j-1) % 14
    d(i,j) = sqrt((row_i - row_j)² + (col_i - col_j)²)

  For CLS token (i=0 or j=0):
    d = 0 (no spatial bias applied)
```

### 3.5 Why This Is Novel

No existing cross-attention mechanism in the literature exploits **guaranteed pixel-level spatial registration** between streams. This is because:

1. In RGB-Depth fusion — depth maps may be misaligned due to sensor offset
2. In CT-PET fusion — different scanners, different resolutions, requires registration
3. In RGB-Thermal — different optics, parallax
4. In text-image fusion — no spatial correspondence at all

**Color deconvolution is unique**: the decomposition is pixel-perfect by construction. The spatial bias is not an approximation — it encodes a mathematical certainty.

This is publishable as a standalone contribution within the architecture.

---

## 4. Bidirectional Cross-Attention — Final Design

### 4.1 Parallel Computation (No Ordering Bias)

Both directions compute from the **original, unmodified** tokens:

$$\hat{\mathbf{H}} = \mathbf{H}_9 + \text{CrossAttn}^{H \leftarrow D}(Q=\mathbf{H}_9,\; K=\mathbf{D}_9,\; V=\mathbf{D}_9;\; B)$$

$$\hat{\mathbf{D}} = \mathbf{D}_9 + \text{CrossAttn}^{D \leftarrow H}(Q=\mathbf{D}_9,\; K=\mathbf{H}_9,\; V=\mathbf{H}_9;\; B)$$

where $B$ is the spatial bias matrix (shared between directions, or separate — an ablation choice).

### 4.2 Feed-Forward After Cross-Attention

Each direction includes its own FFN with residual:

$$\hat{\mathbf{H}} = \text{LN}\left(\hat{\mathbf{H}} + \text{FFN}^H(\hat{\mathbf{H}})\right)$$
$$\hat{\mathbf{D}} = \text{LN}\left(\hat{\mathbf{D}} + \text{FFN}^D(\hat{\mathbf{D}})\right)$$

### 4.3 Internal Module Detail

```
┌──────────────────────────────────────────────────────────┐
│  SPATIALLY-BIASED BIDIRECTIONAL CROSS-ATTENTION BLOCK    │
│                                                          │
│  Inputs: H₉ ∈ ℝ^{B×197×768}                             │
│          D₉ ∈ ℝ^{B×197×768}                             │
│          B  ∈ ℝ^{197×197}     (spatial bias matrix)      │
│                                                          │
│  ┌─── Direction H←D ──────┐  ┌─── Direction D←H ──────┐ │
│  │                        │  │                         │ │
│  │ Q = W_q^{H←D} · LN(H₉)│  │ Q = W_q^{D←H} · LN(D₉)│ │
│  │ K = W_k^{H←D} · LN(D₉)│  │ K = W_k^{D←H} · LN(H₉)│ │
│  │ V = W_v^{H←D} · LN(D₉)│  │ V = W_v^{D←H} · LN(H₉)│ │
│  │                        │  │                         │ │
│  │ logits = QKᵀ/√d + B   │  │ logits = QKᵀ/√d + B    │ │
│  │ attn = softmax(logits) │  │ attn = softmax(logits)  │ │
│  │ out = attn · V         │  │ out = attn · V          │ │
│  │                        │  │                         │ │
│  │ Ĥ = H₉ + W_o^{H←D}·out│  │ D̂ = D₉ + W_o^{D←H}·out│ │
│  │ Ĥ = LN(Ĥ + FFN_H(Ĥ)) │  │ D̂ = LN(D̂ + FFN_D(D̂))  │ │
│  └────────────────────────┘  └─────────────────────────┘ │
│                                                          │
│  Outputs: Ĥ₉ ∈ ℝ^{B×197×768}                            │
│           D̂₉ ∈ ℝ^{B×197×768}                            │
│                                                          │
│  Heads: 12     d_k: 64     FFN hidden: 3072             │
└──────────────────────────────────────────────────────────┘
```

---

## 5. Token Fusion

### 5.1 Gated Fusion (Per-Token)

After blocks 10–12 produce $H_{\text{final}}$ and $D_{\text{final}}$:

**For each patch token** $i \in \{1, \ldots, 196\}$:

$$g_i = \sigma\!\left(W_g \cdot [H_{\text{final},i} \;\|\; D_{\text{final},i}] + b_g\right) \in \mathbb{R}^{768}$$

$$F_i = g_i \odot H_{\text{final},i} + (1 - g_i) \odot D_{\text{final},i}$$

**For the CLS token**:

$$F_0 = W_{\text{cls}} \cdot [\text{CLS}_H \;\|\; \text{CLS}_D] + b_{\text{cls}}$$

This gives a fused token sequence $F \in \mathbb{R}^{B \times 197 \times 768}$.

### 5.2 Why Gated Fusion Is Right Here

The gate $g_i$ is **biologically interpretable**:

| Gate value | Meaning |
|:---|:---|
| $g_i \approx 1$ | Patch $i$ relies on **morphology** (Hematoxylin) |
| $g_i \approx 0$ | Patch $i$ relies on **HER2 signal** (DAB) |
| $g_i \approx 0.5$ | Both stains contribute equally |

**You can visualize these gates as heatmaps** — this gives the paper a strong interpretability figure showing that the model learned biologically meaningful fusion patterns.

---

## 6. Refinement Block

### 6.1 Why It's Needed

Cross-attention **exchanges** information between stains. But the network still needs to **reason over** the fused representation — for example, to aggregate evidence across the entire slide patch (is the membrane staining complete? what fraction of cells are HER2+?).

A single transformer block after fusion provides this capacity.

### 6.2 Design

```
F (197×768) — fused tokens
    │
    ▼
LayerNorm
    │
    ▼
Multi-Head Self-Attention (12 heads, d=768)
    │
    + residual
    │
    ▼
LayerNorm
    │
    ▼
FFN (768 → 3072 → 768)
    │
    + residual
    │
    ▼
F_refined (197×768)
```

**Initialization**: Xavier uniform (trained from scratch, not pretrained). This is intentional — this block learns to reason over *fused stain representations*, which have no counterpart in ImageNet.

**Parameter cost**: ~7.1M parameters (identical to one ViT-B block).

---

## 7. Classification Head

```
F_refined (197×768)
    │
    ├── CLS token: F_refined[:, 0, :]           → cls  ∈ ℝ^{B×768}
    │
    └── Patch tokens: mean(F_refined[:, 1:, :]) → gap  ∈ ℝ^{B×768}
    │
    ▼
z = [cls ‖ gap]  ∈ ℝ^{B×1536}
    │
    ▼
LayerNorm(1536)
    │
    ▼
Linear(1536, 768) → GELU → Dropout(0.1)
    │
    ▼
Linear(768, 4)
    │
    ▼
logits ∈ ℝ^{B×4}  →  {0, 1+, 2+, 3+}
```

> [!NOTE]
> The classification head uses **both CLS and GAP**. CLS captures global summary; GAP captures average spatial evidence. This dual pooling is more robust than either alone and is standard in recent ViT classification papers.

---

## 8. Complete Tensor Flow

```
Step 0: Input
        x_rgb ∈ ℝ^{B × 3 × 224 × 224}

Step 1: Color Deconvolution (fixed, no gradient)
        M_stain = Ruifrok H-DAB stain matrix (3×3, fixed)
        OD = -log₁₀(x_rgb / 255 + ε)         ∈ ℝ^{B × 3 × 224 × 224}
        [H, DAB, Res] = M_stain⁻¹ · OD       (per-pixel matrix multiply)
        x_H ∈ ℝ^{B × 1 × 224 × 224}          (Hematoxylin OD channel)
        x_D ∈ ℝ^{B × 1 × 224 × 224}          (DAB OD channel)

Step 2: Learnable Channel Projection (separate weights per stain)
        x_H' = Proj_H(x_H) ∈ ℝ^{B × 3 × 224 × 224}   Conv2d(1, 3, 1×1)
        x_D' = Proj_D(x_D) ∈ ℝ^{B × 3 × 224 × 224}   Conv2d(1, 3, 1×1)

Step 3: Shared Patch Embedding + Positional Encoding
        t_H = PatchEmbed(x_H') + pos_embed  ∈ ℝ^{B × 197 × 768}
        t_D = PatchEmbed(x_D') + pos_embed  ∈ ℝ^{B × 197 × 768}
        (CLS token prepended, shared positional embeddings)

Step 4: Shared Encoder Blocks 1–9 (batched forward pass)
        stacked = cat([t_H, t_D], dim=0)     ∈ ℝ^{2B × 197 × 768}
        for block_i in [1, ..., 9]:
            stacked = Block_i(stacked)
        H₉, D₉ = stacked.chunk(2, dim=0)
        H₉ ∈ ℝ^{B × 197 × 768}
        D₉ ∈ ℝ^{B × 197 × 768}

Step 5: Spatially-Biased Bidirectional Cross-Attention (parallel)
        B = spatial_bias_matrix               ∈ ℝ^{197 × 197}  (learnable)

        # Direction H←D (H queries, DAB provides context)
        Ĥ = H₉ + MHA(Q=LN(H₉), K=LN(D₉), V=LN(D₉), bias=B)
        Ĥ = Ĥ + FFN_H(LN(Ĥ))

        # Direction D←H (DAB queries, H provides context)
        D̂ = D₉ + MHA(Q=LN(D₉), K=LN(H₉), V=LN(H₉), bias=B)
        D̂ = D̂ + FFN_D(LN(D̂))

        Ĥ ∈ ℝ^{B × 197 × 768}
        D̂ ∈ ℝ^{B × 197 × 768}

Step 6: Shared Encoder Blocks 10–12 (batched forward pass)
        stacked = cat([Ĥ, D̂], dim=0)         ∈ ℝ^{2B × 197 × 768}
        for block_i in [10, 11, 12]:
            stacked = Block_i(stacked)
        H_f, D_f = stacked.chunk(2, dim=0)
        H_f ∈ ℝ^{B × 197 × 768}              (H_final)
        D_f ∈ ℝ^{B × 197 × 768}              (D_final)

Step 7: Gated Token Fusion
        # CLS token fusion
        F_0 = Linear([H_f[:,0] ‖ D_f[:,0]])   ∈ ℝ^{B × 768}

        # Patch token fusion (i = 1..196)
        g_i = σ(Linear([H_f[:,i] ‖ D_f[:,i]])) ∈ ℝ^{B × 768}
        F_i = g_i ⊙ H_f[:,i] + (1-g_i) ⊙ D_f[:,i]  ∈ ℝ^{B × 768}

        F = stack([F_0, F_1, ..., F_196])     ∈ ℝ^{B × 197 × 768}

Step 8: Refinement Block (1 transformer block, trained from scratch)
        F_ref = RefinementBlock(F)             ∈ ℝ^{B × 197 × 768}

Step 9: Classification
        cls = F_ref[:, 0, :]                   ∈ ℝ^{B × 768}
        gap = mean(F_ref[:, 1:, :])            ∈ ℝ^{B × 768}
        z   = [cls ‖ gap]                      ∈ ℝ^{B × 1536}
        logits = ClassHead(z)                  ∈ ℝ^{B × 4}
```

---

## 9. Computational Analysis

### 9.1 Parameter Budget

| Component | Parameters | Notes |
|:---|:---|:---|
| Proj_H + Proj_D | 18 | 2 × Conv2d(1, 3, 1×1) — 9 params each |
| Shared ViT-B/16 (blocks 1–12) | 86.0M | 100% shared, no adapters |
| Spatial bias matrix $B$ | 38,809 | 197×197, learnable |
| Cross-attn QKV+Proj (×2 directions) | 4.7M | 2 × (4 × Linear(768, 768)) |
| Cross-attn FFN (×2 directions) | 4.7M | 2 × (768→3072→768) |
| Gated fusion | 1.2M | Linear(1536, 768) + bias |
| Refinement block | 7.1M | 1 standard transformer block |
| Classification head | 1.2M | 1536→768→4 |
| **Total** | **~105M** | |

### 9.2 Overhead vs. Baseline

| Metric | Baseline ViT-B/16 | DSCA-ViT | Δ |
|:---|:---|:---|:---|
| Parameters | 86M | ~105M | **+22%** |
| FLOPs | 17.6G | ~38G | **+116%** |
| GPU Memory (B=32) | ~6.5 GB | ~14 GB | **+115%** |

> [!NOTE]
> The parameter increase is **22%**, not 100%. The FLOP doubling is inherent to any dual-stream design, but the two encoder passes are **batched** into a single forward pass (Step 4 & 6), so wall-clock overhead on GPU is typically **~40-60%**, not 100%.

### 9.3 The Parameter Breakdown Tells a Story

```
Shared encoder:   86.0M  (82%)  ← pretrained, provides the heavy lifting
Cross-attention:   9.4M  ( 9%)  ← the novel contribution
Refinement:        7.1M  ( 7%)  ← post-fusion reasoning
Fusion + Head:     2.4M  ( 2%)  ← lightweight
Projections:       ~39K  ( 0%)  ← negligible
```

Only **~18%** of the parameters are new. The rest are pretrained. This is a strong argument for reviewers who worry about data efficiency.

---

## 10. Training Strategy

| Aspect | Setting |
|:---|:---|
| **Pretrained weights** | ImageNet ViT-B/16 → shared encoder (blocks 1–12) |
| **New components** | Projections, cross-attention, refinement block, fusion, head — all Xavier init |
| **Freeze schedule** | Freeze shared encoder for first 5 epochs; unfreeze with LR×0.1 |
| **LR: shared encoder** | 1×10⁻⁵ (after unfreeze) |
| **LR: new components** | 1×10⁻⁴ |
| **Optimizer** | AdamW (β₁=0.9, β₂=0.999, wd=0.05) |
| **Schedule** | Cosine decay with 5-epoch linear warmup |
| **Augmentation** | Applied to RGB **before** deconvolution (consistent transforms) |
| **Augmentation types** | RandomHorizontalFlip, RandomVerticalFlip, RandomRotation(90°), ColorJitter(brightness=0.1, contrast=0.1) — keep color jitter mild to not distort stain signals |
| **Loss** | Cross-entropy (start here); ordinal regression as ablation |
| **Batch size** | 32 (effective batch = 32 images, 64 stain inputs via batching) |
| **Epochs** | 50–100 with early stopping (patience=10) |

> [!TIP]
> **Data augmentation must happen on RGB before deconvolution.** If you augment H and DAB independently, you break the spatial correspondence between them. Both channels must see the same geometric transform.

---

## 11. Ablation Plan (Streamlined)

### Core Ablations — Architecture Justification

| ID | Experiment | What It Proves |
|:---|:---|:---|
| **A1** | Baseline ViT-B/16 (RGB) | Baseline to beat |
| **A2** | H-only → ViT → classify | Morphology alone is insufficient |
| **A3** | DAB-only → ViT → classify | HER2 signal alone lacks context |
| **A4** | [H, DAB, 0] as 3-channel input, single ViT | Stain separation helps even without dual-stream |
| **A5** | Dual-stream, NO cross-attention, CLS concat | Dual-stream alone (no interaction) |
| **A6** | **Full DSCA-ViT** (proposed) | Complete architecture |

### Cross-Attention Ablations

| ID | Experiment | What It Proves |
|:---|:---|:---|
| **B1** | Unidirectional H←DAB only | Whether bidirectional is needed |
| **B2** | Unidirectional DAB←H only | Which direction is more important |
| **B3** | Bidirectional, CLS-only (not full-token) | Whether token-level attention matters |
| **B4** | Bidirectional, **no spatial bias** (S0) | Whether spatial bias helps |
| **B5** | Bidirectional, **Gaussian prior** (S2) | Fixed vs. learnable bias |
| **B6** | Bidirectional, **hard mask** (S3, window=3) | Whether strict locality helps |

### Placement Ablations

| ID | After Layer | What It Proves |
|:---|:---|:---|
| **C1** | 4 | Early fusion |
| **C2** | 6 | Mid fusion |
| **C3** | 9 (default) | Late-mid fusion |
| **C4** | 11 | Very late fusion |

### Refinement Block Ablation

| ID | Experiment | What It Proves |
|:---|:---|:---|
| **D1** | No refinement block (classify directly after fusion) | Whether post-fusion reasoning helps |
| **D2** | 1 refinement block (default) | Proposed design |

> [!TIP]
> **Priority order for ablations**: A1–A6 first (proves the architecture), then B1–B6 (proves the attention design), then C1–C4 (placement sensitivity), then D1–D2 (refinement). If time-constrained, A-series and B-series are non-negotiable for publication.

---

## 12. Publication Novelty — The Story

### The Narrative Arc

```
1. "HER2 IHC images are NOT natural images."
   → Biological motivation

2. "They consist of two overlapping stains with different biological roles."
   → Color deconvolution (domain knowledge)

3. "We process them as separate streams through a shared ViT."
   → Dual-stream, parameter-efficient design

4. "We fuse them using bidirectional cross-attention."
   → Rich token-level interaction

5. "We bias attention toward corresponding spatial positions,
    because the streams are pixel-aligned by construction."
   → Core technical novelty (spatially-biased cross-attention)

6. "Gate values reveal which stain drives classification at each location."
   → Built-in interpretability
```

### What Makes This Publishable

| Claim | Evidence Needed |
|:---|:---|
| Stain decomposition improves ViT classification | A1 vs. A6 |
| Dual-stream outperforms single-stream decomposition | A4 vs. A6 |
| Bidirectional outperforms unidirectional | B1, B2 vs. A6 |
| Token-level attention outperforms CLS-only | B3 vs. A6 |
| Spatial bias improves cross-attention | B4 vs. A6 |
| Gate values are biologically interpretable | Visualization figures |

### Key Figures for the Paper

1. **Architecture diagram** (Section 2)
2. **Cross-attention heatmap**: Show that patch $i$ in H attends most strongly to patch $i$ in DAB (validates the spatial bias hypothesis)
3. **Gate value heatmap**: Overlay $g_i$ on original image — expect DAB-dominant gates on membrane regions, H-dominant gates on nuclear clusters
4. **Attention map comparison**: Standard cross-attention vs. spatially-biased — show the bias produces more focused, biologically meaningful patterns
5. **Per-class confusion matrix**: Especially 2+ vs. 3+ (the hardest clinical distinction)

---

## 13. Open Questions

> [!IMPORTANT]
> **Q1 — Stain Vectors**: Will you use the standard Ruifrok H-DAB matrix, or do you want to estimate stain vectors per slide (Macenko)? Standard is simpler and more reproducible.

> [!IMPORTANT]
> **Q2 — Dataset Size & Class Balance**: How many images in HER2-IHC-40x, and what is the class distribution? This determines whether we need class-weighted loss or oversampling.

> [!IMPORTANT]
> **Q3 — Publication Venue**: Medical imaging (MICCAI, MedIA, TMI) or computer vision (CVPR workshop, ECCV)? This affects how we frame the narrative.

> [!IMPORTANT]
> **Q4 — Spatial Bias Strategy**: I recommend starting with S1 (learnable additive bias). Do you agree, or do you prefer the cleaner theoretical argument of S2 (fixed Gaussian)?
