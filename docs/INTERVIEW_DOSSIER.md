# NexoraLM Technical Interview Dossier

## 1. Architecture Fundamentals
- **Why decoder-only?**
  Autoregressive decoder-only models simplify next-token prediction, allow causal KV-caching during generation, and eliminate cross-attention parameter overhead while maintaining strong general few-shot and zero-shot reasoning capabilities.
- **Why 16 layers and 768 hidden dimension?**
  With $d_{\text{model}} = 768$, head dimension $d_h = 64$ fits 12 query heads cleanly. 16 layers provide sufficient hierarchical depth for semantic abstraction while keeping the total parameter count at ~125.85M, fitting easily into 16 GB T4 VRAM.
- **Why GQA (12 Q heads, 4 KV heads)?**
  In standard MHA, KV cache memory scales with $n_q \times d_h$. With GQA ($n_{kv} = 4$), the KV cache memory footprint is reduced by a factor of $12 / 4 = 3\times$ (66.7% memory reduction), enabling longer batch generation under tight VRAM constraints without the representational collapse of single-head MQA.
- **Why SwiGLU?**
  SwiGLU computes $(\text{swish}(x W_{gate})) \odot (x W_{up}) W_{down}$. The multiplicative gating allows non-linear modulation of feature dimensions, consistently outperforming standard ReLU or GELU in validation perplexity per parameter.
- **Why RMSNorm over LayerNorm?**
  RMSNorm normalizes by root-mean-square without mean centering: $\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum x_i^2 + \epsilon}} \odot \gamma$. This eliminates the mean-computation and subtraction passes, yielding 10-50% faster kernel execution while retaining identical numerical stability.
- **Why RoPE (Rotary Position Embeddings)?**
  RoPE encodes position by rotating pairs of coordinates in the complex plane using orthogonal rotation matrices: $R_{\Theta, m}^d x_m$. Attention scores between position $m$ and $n$ depend purely on relative distance $m-n$, providing natural relative position awareness with zero additive embedding parameters.
- **Why Pre-Normalization?**
  Pre-LN applies normalization before the attention and FFN blocks ($x + f(\text{RMSNorm}(x))$), creating an unhindered gradient highway through the residual stream, preventing vanishing or exploding gradients at layer 16 without needing warmup-dependent learning rate tricks required by Post-LN.
- **Why Tied Embeddings?**
  Tying input embedding weights with the final LM head projection ($W_{lm\_head} = W_{embed}^T$) saves $32,768 \times 768 \approx 25.17\text{M}$ parameters (20% of total parameters). For a 126M parameter model, this keeps the model compact while sharing semantic representations between token ingestion and output generation.
- **Why exact 32,768 vocabulary?**
  32,768 is $2^{15}$, optimizing memory alignment and GPU tensor core warp tile sizes, and provides broad byte-level subword coverage for English and code while preventing embedding parameter bloat.

## 2. Pretraining Dynamics
- **Why FineWeb-Edu?**
  FineWeb-Edu filters web text with educational quality classifiers, providing high token density for logic, factual reasoning, and clear syntax compared to uncurated raw web scrapes.
- **Why 300M tokens?**
  Given an NVIDIA T4 single-GPU constraint on Kaggle, 300M tokens provides a feasible multi-hour training budget (~2.38 tokens/param) that fully validates the loss convergence curve, checkpointing, and downstream adaptation lifecycle.
- **Why AdamW with $\beta_2 = 0.95$?**
  AdamW decouples weight decay from the gradient update step ($w_{t+1} = w_t - \eta_t \lambda w_t - \eta_t \frac{m_t}{\sqrt{v_t} + \epsilon}$). Using $\beta_2 = 0.95$ instead of $0.999$ reduces the memory horizon of second-moment variance tracking, responding faster to non-stationary gradient dynamics common in LLM pretraining.
- **Why Gradient Accumulation?**
  Micro-batch size 4 fits easily into 16 GB VRAM with FP16 activations. 16 accumulation steps yield an effective batch size of $4 \times 16 \times 2048 = 131,072$ tokens (~131k tokens), providing stable gradient estimates.

## 3. Supervised Fine-Tuning (SFT)
- **Why Assistant-Only Loss?**
  During SFT, the prompt (system instructions and user queries) is context, not generated content. Masking prompt tokens to $-100$ in cross-entropy ensures the model only learns the conditional probability distribution $P(\text{Assistant} \mid \text{System}, \text{User})$. Penalizing prompt tokens wastes model capacity on memorizing input questions.

## 4. Parameter-Efficient Adaptation (LoRA & QLoRA)
- **Mathematical Form of LoRA:**
  For frozen weight $W_0 \in \mathbb{R}^{d_{out} \times d_{in}}$, adaptation is:
  $$W' = W_0 + \frac{\alpha}{r} B A, \quad A \in \mathbb{R}^{r \times d_{in}}, B \in \mathbb{R}^{d_{out} \times r}$$
  where $A \sim \mathcal{N}(0, \sigma^2)$ and $B = 0$ at initialization.
- **Parameter Calculation:**
  For rank $r=16$ on $Q, K, V, O$:
  - $Q: 16 \times (768 + 768) = 24,576$
  - $K: 16 \times (768 + 256) = 16,384$
  - $V: 16 \times (768 + 256) = 16,384$
  - $O: 16 \times (768 + 768) = 24,576$
  - Per layer: $81,920$ params.
  - Across 16 layers: $16 \times 81,920 = 1,310,720$ parameters (~1.31M params, ~1.04% of total base model).
- **QLoRA vs LoRA:**
  LoRA stores base weights in FP16 (2 bytes/weight). QLoRA quantizes base weights into 4-bit (0.5 bytes/weight), dequantizing on-the-fly during forward pass to compute activations while backpropagating gradients strictly into FP32/FP16 adapter matrices.

## 5. Direct Preference Optimization (DPO)
- **Objective:**
  $$\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \left( \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right) \right]$$
- **Role of $\beta$:**
  $\beta$ acts as the inverse temperature of the implicit reward model. A high $\beta$ penalizes deviations from $\pi_{\text{ref}}$ heavily; a low $\beta$ allows aggressive policy divergence, risking mode collapse or degenerative repetitive outputs.
- **Reference Model Freezing:**
  $\pi_{\text{ref}}$ is initialized as a frozen snapshot of `NexoraLM-SFT`. It serves as the implicit KL-divergence constraint anchor.

## 6. KV-Cache & Inference Complexity
- **Time Complexity:**
  Without KV cache: Generating token $T$ requires recalculating attention across all previous tokens $1 \dots T-1$, leading to $O(T^2)$ computations per step and $O(T^3)$ total generation cost.
  With KV cache: Previous Key and Value representations are cached in memory; only the new token Query is computed, yielding $O(T)$ attention computation per step and $O(T^2)$ total generation cost.
- **KV Cache Memory Size:**
  For batch size $B$, context length $T$, layers $L=16$, $n_{kv}=4$, $d_h=64$:
  $$\text{Memory} = 2 \times L \times B \times n_{kv} \times T \times d_h \times \text{bytes\_per\_elem}$$
  At FP16 (2 bytes), for $B=1, T=2048$:
  $$\text{Memory} = 2 \times 16 \times 1 \times 4 \times 2048 \times 64 \times 2 = 33,554,432 \text{ bytes} = 32 \text{ MB}$$
  (With standard MHA $n_q=12$, this would be $96 \text{ MB}$, proving the 66.7% memory savings of GQA).

## 7. Tensor-Shape Dossier
| Stage | Tensor Representation | Shape | Dimensional Rationale |
|-------|----------------------|-------|------------------------|
| Input | Token IDs | $[B, T]$ | Batch of token sequences up to context window $T=2048$ |
| Embedding | Token Embeddings | $[B, T, 768]$ | Ingested through $32,768 \times 768$ weight table |
| Query Projection | $Q$ | $[B, 12, T, 64]$ | $12$ query heads of head dimension $64$ |
| Key Projection | $K$ | $[B, 4, T, 64]$ | $4$ key heads for Grouped Query Attention |
| Value Projection | $V$ | $[B, 4, T, 64]$ | $4$ value heads matching key heads |
| GQA Expansion | $K_{\text{exp}}, V_{\text{exp}}$ | $[B, 12, T, 64]$ | Repeated 3x across head dimension ($12/4 = 3$) |
| Attention Scores | $S = Q K_{\text{exp}}^T / \sqrt{64}$ | $[B, 12, T, T]$ | Scaled dot-product pairwise causal attention matrix |
| Attention Output | $O = \text{softmax}(S) V_{\text{exp}}$ | $[B, T, 768]$ | Transposed and reshaped after output projection $W_o$ |
| SwiGLU Gating | $W_{\text{gate}}(x), W_{\text{up}}(x)$ | $[B, T, 2048]$ | Intermediate projection to $2048$ dimensions |
| SwiGLU Output | $\text{SiLU}(\text{Gate}) \odot \text{Up} \cdot W_{\text{down}}$ | $[B, T, 768]$ | Projected back to residual dimension $768$ |
| Final Normalization | $\text{RMSNorm}(x)$ | $[B, T, 768]$ | Pre-head root-mean-square normalization |
| LM Head Logits | $\text{Logits}$ | $[B, T, 32768]$ | Unnormalized next-token probability distribution |

## 8. Parameter Count Breakdown
- **Embedding Table (Tied)**: $32,768 \times 768 = 25,165,824$
- **Attention (per layer)**:
  - $W_q$: $768 \times 768 = 589,824$
  - $W_k$: $768 \times 256 = 196,608$
  - $W_v$: $768 \times 256 = 196,608$
  - $W_o$: $768 \times 768 = 589,824$
  - Layer Total: $1,572,864$
- **SwiGLU FFN (per layer)**:
  - $W_{\text{gate}}$: $768 \times 2048 = 1,572,864$
  - $W_{\text{up}}$: $768 \times 2048 = 1,572,864$
  - $W_{\text{down}}$: $2048 \times 768 = 1,572,864$
  - Layer Total: $4,718,592$
- **RMSNorms (per layer)**: $2 \times 768 = 1,536$
- **Total per layer**: $1,572,864 + 4,718,592 + 1,536 = 6,292,992$
- **16 Layers Total**: $16 \times 6,292,992 = 100,687,872$
- **Final RMSNorm**: $768$
- **Grand Total**: $25,165,824 + 100,687,872 + 768 = 125,854,464$ (~125.85M parameters).
