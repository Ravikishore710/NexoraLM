# NexoraLM Architecture Specification

## Architecture Overview
NexoraLM-126M is a decoder-only Transformer built from first principles with modern design choices:
- Pre-Layer Normalization using RMSNorm.
- Rotary Position Embeddings (RoPE) applied to Query and Key projections.
- Grouped Query Attention (GQA) with 12 Query heads and 4 Key/Value heads.
- SwiGLU Feed-Forward Networks.
- Weight tying between token embedding and LM head projection.
- Bias-free linear layers.

## Dimensions & Hyperparameters
- Layers ($L$): 16
- Model dimension ($d_{\text{model}}$): 768
- Query attention heads ($n_q$): 12
- Key/Value attention heads ($n_{kv}$): 4
- Head dimension ($d_h$): 64 ($12 \times 64 = 768$, $4 \times 64 = 256$)
- Intermediate FFN dimension ($d_{ffn}$): 2048
- RMSNorm epsilon: $1.0\text{e}-5$
- RoPE theta base: 10000.0
- Context window: 2048
- Vocabulary size ($V$): 32,768 (strictly including all special tokens)

## Parameter Derivation
1. Token Embedding / LM Head (tied):
   $V \times d_{\text{model}} = 32,768 \times 768 = 25,165,824$ (~25.17M)

2. Attention per layer:
   - Query projection ($W_q$): $768 \times 768 = 589,824$
   - Key projection ($W_k$): $768 \times 256 = 196,608$
   - Value projection ($W_v$): $768 \times 256 = 196,608$
   - Output projection ($W_o$): $768 \times 768 = 589,824$
   - Subtotal per layer: $1,572,864$ (~1.57M)

3. FFN per layer (SwiGLU):
   - Gate projection ($W_{gate}$): $768 \times 2048 = 1,572,864$
   - Up projection ($W_{up}$): $768 \times 2048 = 1,572,864$
   - Down projection ($W_{down}$): $2048 \times 768 = 1,572,864$
   - Subtotal per layer: $4,718,592$ (~4.72M)

4. Normalization per layer:
   - Attention RMSNorm: $768$
   - FFN RMSNorm: $768$
   - Subtotal per layer: $1,536$

5. Final RMSNorm:
   - $768$

6. Total Parameters:
   $\text{Per-layer} = 1,572,864 + 4,718,592 + 1,536 = 6,292,992$
   $\text{16 layers} = 16 \times 6,292,992 = 100,687,872$
   $\text{Total} = 25,165,824 + 100,687,872 + 768 = 125,854,464 \approx 125.85\text{M} \approx 126\text{M}$

## Forward Pass Tensor Shapes
- Input IDs: $[B, T]$
- Token Embeddings: $[B, T, 768]$
- Query ($Q$): $[B, 12, T, 64]$
- Key ($K$): $[B, 4, T, 64]$
- Value ($V$): $[B, 4, T, 64]$
- Expanded $K, V$ for GQA (repeated 3x): $[B, 12, T, 64]$
- Attention Scores: $[B, 12, T, T]$
- Attention Output: $[B, T, 768]$
- SwiGLU Output: $[B, T, 768]$
- LM Head Logits: $[B, T, 32768]$
