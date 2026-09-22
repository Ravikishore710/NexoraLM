# NexoraLM

NexoraLM is a compact, custom 126M-parameter decoder-only language model built from first principles in PyTorch and carried through the complete modern lifecycle:
Tokenizer -> Architecture -> Micro Validation -> Pretraining -> SFT -> LoRA -> QLoRA -> DPO -> Quantization -> KV-Cache Inference -> API Serving.

---

## Architecture Specification
- **Parameters**: 125,854,464 (~126M) with tied input/output embeddings
- **Layers**: 16 Transformer blocks
- **Hidden Dimension**: 768
- **Attention**: Grouped-Query Attention (GQA) with 12 Query heads, 4 Key/Value heads ($3:1$ ratio)
- **Head Dimension**: 64
- **Normalization**: RMSNorm (Pre-LN, $\epsilon = 1\text{e}-5$)
- **Positional Encoding**: Rotary Position Embeddings (RoPE, $\theta = 10000$)
- **Activation**: SwiGLU (intermediate FFN dimension: 2048)
- **Context Length**: 2048 tokens
- **Vocabulary**: Exactly 32,768 (Byte-level BPE with `<UNK>`, `<BOS>`, `<EOS>`, `<PAD>`, `<|system|>`, `<|user|>`, `<|assistant|>`)
- **Linear Biases**: Disabled across all projections

---

## Lifecycle Benchmark Matrix

| Lifecycle Stage | Model Variant | Precision | Parameters (Trainable) | Memory / Weights Size | Throughput / Latency | Empirical Result / Error |
|---|---|---|---|---|---|---|
| **Tokenizer** | Byte BPE | N/A | 32,768 Vocab | 2.3 MB JSON | N/A | FineWeb-Edu trained, exact boundary |
| **Micro Sanity** | NexoraMicro | FP32 | 18.88M (100%) | 2.1 GB VRAM | N/A | Single-batch overfit loss: 0.0054 |
| **Hardware Gate**| NexoraLM-Gate | FP16/SDPA | 125.85M (100%) | 5.85 GB Peak VRAM | 11,042.4 tok/s | Gate Decision: GO (7.55h for 300M) |
| **SFT** | NexoraLM-SFT | FP16 | 125.85M (100%) | 240.0 MB | Assistant-only mask | Loss: 10.43 -> 10.30 (Val PPL: 29.7k) |
| **LoRA** | NexoraLM-LoRA | FP16+Adapter | 1,310,720 (1.03%) | ~5.2 MB Adapter | N/A | Target: `q, k, v, o` ($r=16, \alpha=32$) |
| **QLoRA** | NexoraLM-QLoRA| 4-bit+Adapter | 1,310,720 (1.03%) | 64.7 MB Base + 5.2 MB | N/A | 4-bit base weights, FP16 adapters |
| **DPO** | NexoraLM-DPO | FP16 | 125.85M (Policy) | 240.0 MB | Policy + Frozen Ref | Loss: 0.6931 -> 0.3956, Margin: +0.7260 |
| **Quant (INT8)**| NexoraLM-INT8 | INT8 | Static Quantized | 120.55 MB (50.2%) | Fast Linear GEMM | Mean Relative Error: **0.0078** (< 0.8%) |
| **Quant (INT4)**| NexoraLM-INT4 | INT4 | Static Quantized | 64.69 MB (27.0%) | 3.7x compression | Mean Relative Error: **0.1005** (~10.0%) |
| **Serving** | FastAPI Engine| FP16/INT8 | 125.85M | Production Server | SSE Streaming | `/health`, `/model`, `/v1/chat/completions` |

---

## Repository Layout
```
nexoralm/
├── src/
│   ├── tokenizer/      # Byte-level BPE tokenizer (32,768 vocab, assistant masking)
│   ├── model/          # 126M Decoder-only Transformer & tied embeddings
│   ├── attention/      # Grouped Query Attention (GQA) & iterative KV-cache
│   ├── rope/           # Rotary Position Embeddings
│   ├── normalization/  # RMSNorm
│   ├── ffn/            # SwiGLU feed-forward network
│   ├── training/       # AdamW, cosine schedule, FP16 GradScaler, checkpoints
│   ├── lora/           # Custom LoRA implementation & weight merging
│   ├── dpo/            # Custom DPO loss and implicit preference reward margins
│   ├── quantization/   # INT8 per-channel & INT4 group-wise quantization
│   ├── inference/      # Autoregressive generation (greedy, top-p, top-k, temp)
│   └── serving/        # FastAPI inference server with SSE streaming
├── configs/            # Parameterized YAML configs for all training phases
├── scripts/            # Pipeline runners (pretrain, sft, lora, qlora, dpo, quantize, serve)
├── tests/              # 22 automated unit tests covering math, shapes, masks, and serving
├── docs/               # 14 engineering and interview documentation files
└── outputs/            # Empirical execution logs, benchmarks, and evaluation reports
```

---

## Quickstart Guide

### 1. Run Automated Unit Tests
```bash
pytest tests/ -v
```
All 22 unit tests validate causal attention gradient isolation, KV cache numerical equivalence, parameter counts, LoRA weight merging, DPO loss, and INT8/INT4 quantization fidelity.

### 2. Post-Training Quantization Benchmark
```bash
python scripts/quantize.py --model_path checkpoints/nexoralm_base.pt
```

### 3. Parameter-Efficient Fine-Tuning (LoRA & QLoRA)
```bash
# LoRA Benchmark
python scripts/lora.py --base_model_path checkpoints/nexoralm_base.pt

# QLoRA Benchmark (4-bit base weights)
python scripts/qlora.py
```

### 4. Direct Preference Optimization (DPO)
```bash
python scripts/dpo.py --sft_checkpoint checkpoints/nexoralm_sft.pt --steps 3
```

### 5. Start OpenAI-Compatible API Server
```bash
python scripts/serve.py --checkpoint checkpoints/nexoralm_aligned.pt --port 8000
```

### 6. Interactive Terminal Chat
```bash
python scripts/chat_cli.py --url http://localhost:8000
```

---

## Pretraining on Kaggle GPU
The 300M-token pretraining job runs on NVIDIA Tesla T4:
```bash
python scripts/kaggle_runner.py status --slug nexoralm-300m-pretraining
python scripts/kaggle_runner.py output --slug nexoralm-300m-pretraining --out outputs/pretrain_300m_final
```
