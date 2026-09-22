# Architecture Decision Records (ADR)

## ADR-001: Decoder-Only Architecture
- Date: 2026-09-21
- Decision: Implement an autoregressive decoder-only Transformer instead of encoder-decoder or prefix-LM.
- Rationale: Standardizes causal generative modeling, enables direct KV-cache inference, and aligns with modern production LLM paradigms.
- Status: Accepted.

## ADR-002: Grouped Query Attention (GQA) with 12 Q Heads and 4 KV Heads
- Date: 2026-09-21
- Decision: Use 12 query heads and 4 key/value heads rather than Multi-Head Attention (12 KV) or Multi-Query Attention (1 KV).
- Rationale: Reduces KV-cache memory by 66.7% compared to MHA while retaining head diversity superior to MQA.
- Status: Accepted.

## ADR-003: SwiGLU Activation Function
- Date: 2026-09-21
- Decision: Use SwiGLU instead of GELU or standard ReLU.
- Rationale: Empirically improves perplexity and gradient flow in modern LLMs at equivalent parameter counts.
- Status: Accepted.

## ADR-004: Pre-Layer Normalization with RMSNorm
- Date: 2026-09-21
- Decision: Use RMSNorm in Pre-LN configuration without mean centering.
- Rationale: 10-50% computational speedup over standard LayerNorm while offering identical training stability.
- Status: Accepted.

## ADR-005: Tied Input/Output Embeddings
- Date: 2026-09-21
- Decision: Tie token embedding weights to the LM head projection.
- Rationale: Saves ~25.17M parameters, keeping the parameter budget at ~126M without sacrificing representation quality.
- Status: Accepted.

## ADR-006: Exact 32,768 Vocabulary Boundary
- Date: 2026-09-21
- Decision: Set total vocabulary size strictly to 32,768, including all special and chat tokens.
- Rationale: Preserves exact powers of 2 for memory alignment and keeps embedding parameter count fixed at 25,165,824.
- Status: Accepted.

## ADR-007: DPO Reference Model Sourced from Frozen SFT
- Date: 2026-09-21
- Decision: Fork NexoraLM-SFT into trainable policy and frozen reference for DPO.
- Rationale: Guarantees exact reference log-probability grounding without training a separate reward model.
- Status: Accepted.

## ADR-008: Fused Scaled Dot-Product Attention (SDPA)
- Date: 2026-09-21
- Decision: Utilize PyTorch `F.scaled_dot_product_attention` for pretraining forward passes, falling back to explicit attention during step-by-step KV-cached generation.
- Rationale: Prevents quadratic materialization of $[B, H, T, T]$ score tensors in VRAM, enabling 2048-token pretraining sequences on 16GB accelerators.
- Status: Accepted.

## ADR-009: Micro-Batch Size 2 with 32 Gradient Accumulation Steps
- Date: 2026-09-21
- Decision: Configure micro-batch size 2 with 32 gradient accumulation steps.
- Rationale: Preserves the exact 131,072-token effective global batch size while keeping peak activation VRAM comfortably within 16GB GDDR6 capacity.
- Status: Accepted.
