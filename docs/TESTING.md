# NexoraLM Testing Strategy & Verification Plan

## Test Suite Overview
All automated unit tests reside in `tests/` and are executed via `pytest`.

## Target Test Areas
1. `tests/test_shapes.py`:
   - Validates tensor dimensions across RMSNorm, RoPE, GQA attention, SwiGLU FFN, and complete Transformer.
2. `tests/test_attention.py`:
   - Validates causal lower-triangular masking: ensures that mutating future tokens produces zero gradient change on earlier token representations.
3. `tests/test_loss.py`:
   - Tests next-token cross-entropy and assistant-only label masking (-100 target ignore index).
4. `tests/test_kv_cache.py`:
   - Asserts numerical equivalence ($< 1\text{e}-4$ relative tolerance) between full sequence forward pass and incremental autoregressive generation using the KV-cache.
5. `tests/test_lora.py`:
   - Confirms that base model parameters remain frozen while low-rank matrices receive gradients, and verifies weight merging.
6. `tests/test_tokenizer.py`:
   - Verifies exact 32,768 vocabulary boundary, special token encoding/decoding roundtrip fidelity, and chat template formatting.
