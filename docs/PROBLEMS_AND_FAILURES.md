# Problems and Failures Log

## Incident Tracking Table

| Failure ID | Component | Trigger | Root Cause | Resolution | Status |
|------------|-----------|---------|------------|------------|--------|
| FAIL-001   | Auth      | Initial Kaggle API token missing | Kaggle API updated from username/key to access token | Generated and configured Kaggle access token | Resolved |
| FAIL-002   | M2 Sanity | Strict loss equality on CUDA | Floating-point non-determinism in cuBLAS matrix multiplications produced loss delta of ~2e-4 | Assert exact parameter tensor equality (`torch.equal`) and use 1e-3 tolerance for floating point loss comparison | Resolved |
| FAIL-003   | M4 Pretrain | CUDA OOM on Tesla T4 (micro_bs=4, seq_len=2048) | Materializing $[4, 12, 2048, 2048]$ eager attention score matrices across 16 layers required ~12.8 GB activation memory | Adopt PyTorch `F.scaled_dot_product_attention` with `is_causal=True` and adjust `micro_batch_size=2`, `grad_accum=32` to preserve 131k effective token batch | Resolved |
| FAIL-004   | M4 Pretrain | Kaggle script kernel FileNotFoundError | Kaggle script kernels do not include external filesystem files unless mounted via Kaggle datasets | Embedded gzipped base64 tokenizer artifact directly in script with self-extracting bootstrap | Resolved |
| FAIL-005   | M5 / M8 Loading | PyTorch 2.6 `weights_only=True` default | PyTorch 2.6 changed `torch.load` default to `weights_only=True`, rejecting custom dataclass globals | Set `weights_only=False` explicitly for trusted local checkpoints | Resolved |
| FAIL-006   | Checkpoint Keys | State dict key mismatch (`norm1` vs `input_norm`) | Gate script used shorthand layer naming `norm1`/`norm2` whereas modular architecture used `input_norm`/`post_attention_norm` | Executed key realignment script `scripts/align_checkpoint_keys.py` mapping keys seamlessly | Resolved |

## Failure Case Details

### FAIL-001: Kaggle CLI Authentication Format Deprecation
- Date: 2026-09-21
- Component: CLI / Kaggle API
- Symptom: `kaggle kernels list --mine` failed with "Authentication required".
- Root Cause: Legacy `{"username", "key"}` format in `~/.kaggle/kaggle.json` was superseded by Kaggle token-based authentication.
- Resolution: Retrieved new API access token and saved to `~/.kaggle/access_token` and set `KAGGLE_API_TOKEN` environment variable. Verified with successful kernel list query.

### FAIL-002: Floating-Point Non-Determinism in CUDA Checkpoint Verification
- Date: 2026-09-21
- Component: M2 Checkpoint resume verification on Tesla T4
- Symptom: `AssertionError: Checkpoint resume mismatch` during Kaggle run on `assert abs(fresh_loss.item() - loss.item()) < 1e-4`.
- Root Cause: CUDA floating-point reduction order variations in GPU matrix multiplication created slight numerical deviation (~2e-4) between separate forward passes.
- Resolution: Validated checkpoint state restoration using parameter bitwise equality (`torch.equal(p1, p2)`) across all weights, with 1e-3 threshold on loss.

### FAIL-003: CUDA Out of Memory on Tesla T4 during 2048-Token Pretraining Gate
- Date: 2026-09-21
- Component: M4 Pretraining Throughput Gate
- Symptom: `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 768.00 MiB` on Tesla T4 (14.56 GB allocated memory).
- Root Cause: Eager quadratic attention matrix calculation materialized $[4, 12, 2048, 2048]$ floats per layer, requiring ~800 MB per layer and ~12.8 GB total activation memory across 16 layers.
- Resolution: Refactored attention to use `F.scaled_dot_product_attention` (which utilizes memory-efficient fused kernels without quadratic activation buffers) and set `micro_batch_size=2` with `grad_accum=32`, keeping the effective batch size invariant at 131,072 tokens while dropping activation VRAM to under 6 GB.

### FAIL-004: Kaggle Script Kernel FileNotFoundError for Local Tokenizer
- Date: 2026-09-22
- Component: Kaggle 300M Pretraining Kernel
- Symptom: `FileNotFoundError: tokenizer.json not found in kernel root` immediately on kernel launch.
- Root Cause: Kaggle script kernels only receive the script payload (`main.py`) by default; auxiliary workspace files are not automatically mirrored unless packaged as datasets.
- Resolution: Serialized and compressed `checkpoints/tokenizer.json` into a compact base64 gzip payload (533 KB) directly inside `main.py` with an automatic self-extracting bootstrap before execution.

### FAIL-005: PyTorch 2.6 Weights-Only Deserialization Policy Rejection
- Date: 2026-09-22
- Component: Checkpoint Deserialization (`torch.load`) in SFT and DPO
- Symptom: `_pickle.UnpicklingError: Weights only load failed. Unsupported global: GLOBAL src.model.transformer.NexoraConfig`.
- Root Cause: PyTorch 2.6 changed the default `weights_only` argument in `torch.load` from `False` to `True`, refusing to deserialize custom classes or dataclasses without explicit allowlisting.
- Resolution: Explicitly passed `weights_only=False` to `torch.load` for verified local project checkpoints.

### FAIL-006: State Dict Key Mismatch Between Gate Script and Modular Architecture
- Date: 2026-09-22
- Component: Checkpoint Loading in Downstream Stages
- Symptom: `RuntimeError: Missing key(s) in state_dict: "layers.0.input_norm.weight"... Unexpected key(s): "layers.0.norm1.weight"`.
- Root Cause: Standalone gate script used abbreviated module names `norm1` and `norm2` whereas modular architecture in `src/` used standard LLaMA-style naming `input_norm` and `post_attention_norm`.
- Resolution: Implemented key alignment migration script (`scripts/align_checkpoint_keys.py`) translating keys seamlessly into `checkpoints/nexoralm_base.pt`.

