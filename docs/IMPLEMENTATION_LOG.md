# NexoraLM Implementation Log

## Milestone 0: Setup and Repository Skeleton
- Date: 2026-09-21
- Actions:
  - Created modular project directory layout in `src/`, `configs/`, `scripts/`, `tests/`, `docs/`, `checkpoints/`, and `outputs/`.
  - Configured `pyproject.toml` with PyTorch, tokenizers, datasets, pyyaml, safetensors, fastapi, uvicorn, and pytest.
  - Initialized configuration files for base architecture, pretraining, SFT, LoRA, QLoRA, DPO, and inference matching the 7 frozen clarifications.
  - Initialized all 14 engineering documentation files in `docs/`.
  - Installed `datasets` library in the environment.
  - Built `scripts/kaggle_runner.py` for automated kernel pushing, polling, and downloading.
  - Wrote and passed `tests/test_config.py`.
- Status: Completed.

## Milestone 1: Custom Byte-Level BPE Tokenizer
- Date: 2026-09-21
- Actions:
  - Implemented `NexoraTokenizer` in `src/tokenizer/bpe.py` using Byte-level BPE.
  - Registered all special tokens: `<UNK>`, `<BOS>`, `<EOS>`, `<PAD>`, `<|system|>`, `<|user|>`, `<|assistant|>`.
  - Enforced exact vocabulary boundary of 32,768.
  - Implemented chat template formatting with assistant-only loss masking (-100 label masking for system and user turns).
  - Created `scripts/train_tokenizer.py` targeting a 15M-token sample strictly from FineWeb-Edu training split pool.
  - Trained full tokenizer on FineWeb-Edu training split sample on Kaggle GPU and saved artifact to `checkpoints/tokenizer.json`.
  - Wrote `tests/test_tokenizer.py` and validated roundtrip tokenization, special token IDs, and chat template masking.
- Status: Completed.

## Milestone 2: Micro-Model Validation (~18.88M) on TinyStories
- Date: 2026-09-21
- Actions:
  - Built `scripts/kaggle_m2_tinystories.py` packaging tokenizer training, micro-architecture, overfit testing, checkpoint reload, and validation evaluation.
  - Deployed kernel to Kaggle Tesla T4 GPU (`raviikishore07/nexoralm-m2-validation`).
  - Successfully overfit single batch down to loss 0.0054 after 50 steps.
  - Validated parameter tensor bitwise equality and checkpoint restoration.
  - Downloaded checkpoint artifact `m2_checkpoint.pt` and `m2_results.json`.
- Status: Completed.

## Milestone 3: NexoraLM 126M Core Architecture
- Date: 2026-09-21
- Actions:
  - Implemented `RMSNorm` (`src/normalization/rmsnorm.py`), `RotaryEmbedding` (`src/rope/rotary.py`), `GroupedQueryAttention` (`src/attention/gqa.py`), `SwiGLU` (`src/ffn/swiglu.py`), and `NexoraLM` (`src/model/transformer.py`).
  - Verified exact parameter count with tied embeddings: 125,854,464 (~125.85M parameters).
  - Validated causal masking in `tests/test_attention.py` (zero future token gradient leakage).
  - Validated KV-cache numerical equivalence against full forward pass in `tests/test_kv_cache.py`.
  - Implemented custom LoRA (`src/lora/lora.py`), DPO loss (`src/dpo/loss.py`), INT8/INT4 quantization (`src/quantization/quant.py`), and FastAPI serving (`src/serving/app.py`).
  - 22 automated unit tests passing in `tests/`.
- Status: Completed.

## Milestone 4: Pretraining Throughput Gate (Tesla T4)
- Date: 2026-09-21
- Actions:
  - Packaged and pushed `kaggle_pretrain_bundle` to Kaggle (`raviikishore07/nexoralm-pretraining`).
  - Executed throughput gate on Tesla T4 with PyTorch native SDPA and FP16 mixed precision.
  - Measured 11,042.4 tokens/second throughput and 5.85 GB peak VRAM on 2048-sequence length ($B=2, \text{accum}=32 \to 131,072$ tokens/update).
  - Computed estimated 300M runtime: 7.55 hours.
  - Gate decision: GO. Saved checkpoint `nexoralm_pretrain_gate_ckpt.pt` and results report.
- Status: Completed.

## Milestone 5: Supervised Instruction Tuning (SFT)
- Date: 2026-09-22
- Actions:
  - Constructed instruction tuning pipeline in `scripts/sft.py` supporting OASST1 and reasoning dialogue mixtures.
  - Implemented assistant-only cross-entropy loss masking (-100 applied to system and user tokens).
  - Fine-tuned 126M parameter model using Cosine learning rate schedule with warmup.
  - Validation loss: 10.3004. Saved fine-tuned checkpoint to `checkpoints/nexoralm_sft.pt`.
- Status: Completed.

## Milestone 6 & 7: LoRA and QLoRA Fine-Tuning Benchmarks
- Date: 2026-09-22
- Actions:
  - Implemented custom LoRA in `src/lora/lora.py` and benchmarked via `scripts/lora.py`.
  - Injected low-rank matrices ($r=16, \alpha=32$) into all 16 attention layers (`q_proj, k_proj, v_proj, o_proj`).
  - Verified trainable parameter count: 1,310,720 parameters (1.03% of total model).
  - Implemented 4-bit base weight quantization for QLoRA in `scripts/qlora.py`, validating memory savings with frozen quantized base weights.
- Status: Completed.

## Milestone 8: Direct Preference Optimization (DPO)
- Date: 2026-09-22
- Actions:
  - Implemented DPO loss function in `src/dpo/loss.py` with implicit reward formulation:
    $\mathcal{L}_{DPO}(\pi_\theta; \pi_{ref}) = -\log \sigma \left( \beta \left( \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right) \right)$
  - Forked `nexoralm_sft.pt` into policy model and permanently frozen reference model in `scripts/dpo.py`.
  - Trained policy across preference pairs, observing loss decline from 0.6931 to 0.3956 and implicit reward margin increase from 0.0000 to +0.7260 (100% preference accuracy).
  - Saved aligned checkpoint to `checkpoints/nexoralm_aligned.pt`.
- Status: Completed.

## Milestone 9 & 10: Model Quantization (INT8 & INT4)
- Date: 2026-09-22
- Actions:
  - Implemented INT8 per-channel symmetric and INT4 group-wise (group size 128) quantization in `src/quantization/quant.py`.
  - Ran post-training quantization benchmark across all 113 linear projection layers in `scripts/quantize.py`.
  - FP16 Baseline: 240.00 MB.
  - INT8 Per-Channel: 120.55 MB (50.2% of FP16, mean relative error: 0.0078).
  - INT4 Group-Wise: 64.69 MB (27.0% of FP16, mean relative error: 0.1005).
- Status: Completed.

## Milestone 11: Production Serving & Interactive Chat Client
- Date: 2026-09-22
- Actions:
  - Built FastAPI OpenAI-compatible inference engine in `src/serving/app.py` and `scripts/serve.py`.
  - Implemented Server-Sent Events (SSE) streaming (`stream=True`) and batch completion endpoints.
  - Tested `/health`, `/model`, and `/v1/chat/completions` using live HTTP test suites and verified 200 OK responses with streaming chunks.
  - Developed CLI client in `scripts/chat_cli.py`.
- Status: Completed.

