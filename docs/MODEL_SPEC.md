# NexoraLM Model Specification

## Architectural Invariants
- Base Architecture: Decoder-only autoregressive Transformer.
- Parameter Count: 125.85M parameters (~126M).
- Context Window: 2048 tokens.
- Vocabulary Size: Exactly 32,768 tokens (inclusive of special tokens).
- Weight Tying: Embedding weight matrix is tied to the output language modeling head.
- Attention Style: Grouped Query Attention (GQA).
  - 12 query heads, 4 key-value heads.
  - Query heads per KV head: 3.
  - Head dimension: 64.
- Activation: SwiGLU: $\text{SwiGLU}(x) = (\text{swish}(x W_{gate})) \odot (x W_{up}) W_{down}$.
- Normalization: RMSNorm: $\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2 + \epsilon}} \odot \gamma$, with $\epsilon = 1.0\text{e}-5$.
- Positional Embeddings: Rotary Position Embeddings (RoPE) applied to $Q$ and $K$ heads with base $\theta = 10000.0$.
- Linear Biases: Excluded from all projections.
- Masking: Strict causal lower-triangular mask ensuring $p(x_t \mid x_{<t})$.

## Artifact Family
- NexoraLM-Micro: ~20M validation model for M2.
- NexoraLM-Base: 126M pretrained checkpoint on 300M FineWeb-Edu tokens.
- NexoraLM-SFT: Supervised instruction tuned on 12k combined dialog/reasoning dataset.
- NexoraLM-LoRA: LoRA adapter ($r=16, \alpha=32$) trained on FP16 base.
- NexoraLM-QLoRA: LoRA adapter trained on 4-bit quantized base.
- NexoraLM-DPO: Direct preference optimized checkpoint using frozen SFT reference.
- NexoraLM-Quant: INT8 and INT4 quantized variants for edge/CPU inference.
