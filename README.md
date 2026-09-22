# NexoraLM — 126M Decoder-Only Architecture & Full Systems Lifecycle

<div align="center">

![NexoraLM Architecture](https://img.shields.io/badge/Architecture-126M_Decoder--Only-0284c7?style=for-the-badge&logo=pytorch)
![Grouped-Query Attention](https://img.shields.io/badge/Attention-GQA_(12Q/4KV)-6366f1?style=for-the-badge)
![FineWeb-Edu](https://img.shields.io/badge/Pretraining-300M_Tokens-10b981?style=for-the-badge)
![Tests](https://img.shields.io/badge/Unit_Tests-22%2F22_Passed-emerald?style=for-the-badge)
![License](https://img.shields.io/badge/License-Apache_2.0-blue?style=for-the-badge)

<p align="center">
  <b>A compact, high-efficiency 125.85M parameter language model built from first-principles tensor operations in PyTorch.</b><br>
  Engineered end-to-end: Tokenizer &rarr; GQA Architecture &rarr; 300M Base Pretraining &rarr; SFT &rarr; LoRA &rarr; QLoRA &rarr; DPO &rarr; INT8/INT4 Quantization &rarr; FastAPI SSE Serving & Web Console.
</p>

[**Interactive Web Console**](#interactive-web-console) &bull; [**Architecture Specification**](#architecture-specification) &bull; [**Benchmark Matrix**](#lifecycle-benchmark-matrix) &bull; [**Quickstart**](#quickstart-guide) &bull; [**System Prompts & Outputs**](#system-prompts--prompt-response-demonstration)

</div>

---

## Interactive Web Console

NexoraLM includes a showcase web application (`website/`) featuring an interactive chat interface, Server-Sent Events (SSE) streaming, KV-cache telemetry metrics, and the complete 11-stage lifecycle pipeline.

### Running the Web Application
```bash
# 1. Start the OpenAI-compatible FastAPI backend (optional for live weights)
python scripts/serve.py --checkpoint checkpoints/nexoralm_aligned_300m.pt --port 8000

# 2. In a second terminal, serve the premium frontend
python -m http.server 3000 --directory website
```
Open **`http://localhost:3000`** in your browser to interact with the model console.

---

## System Prompts & Prompt-Response Demonstration

NexoraLM incorporates custom chat templates formatted with `<|system|>`, `<|user|>`, and `<|assistant|>` delimiters, trained with assistant-only loss masking.

### Sample Interaction

**System Prompt:**
```text
<|system|>
You are NexoraLM, a compact, precise, and helpful AI assistant built from scratch.
```

**User Prompt:**
```text
<|user|>
Why should modern language models use Grouped-Query Attention (GQA) instead of standard Multi-Head Attention (MHA)?
```

**NexoraLM Response:**
```text
<|assistant|>
Grouped-Query Attention (GQA) shares key and value projection heads across multiple query heads. 
In NexoraLM, we configure 12 Query heads and 4 Key/Value heads (a 3:1 sharing ratio). 

This design delivers two major advantages:
1. Memory Bandwidth Reduction: During autoregressive decoding, memory bandwidth is the primary bottleneck. GQA reduces the size and memory access overhead of the KV-cache by 66.7%.
2. Capacity Preservation: Unlike Multi-Query Attention (MQA) which compresses down to a single KV head, GQA retains multiple distinct KV subspaces, preserving expressiveness close to standard MHA with a fraction of the memory footprint.
```

---

## Architecture Specification

| Component | Parameter / Choice | Engineering Rationale |
|:---|:---|:---|
| **Parameter Count** | **125,854,464** (~126M) | Optimal scale for complete local lifecycle research & compute-bound execution |
| **Layer Depth** | 16 Transformer Blocks | Balanced depth-to-width ratio ensuring stable gradient flow |
| **Hidden Dimension** | 768 ($d_{\text{model}}$) | Standard embedding representation dimension |
| **Attention Mechanism** | Grouped-Query Attention (GQA) | 12 Query heads, 4 Key/Value heads ($3:1$ ratio, $d_{\text{head}} = 64$) |
| **KV-Cache Footprint** | 66.7% reduction vs MHA | Optimized for high-throughput iterative autoregressive decoding |
| **Normalization** | RMSNorm (Pre-LN, $\epsilon = 1\text{e}-5$) | Removes mean-centering overhead; ~7% faster throughput than LayerNorm |
| **Positional Embeddings** | Rotary Position Embeddings (RoPE) | Applied directly to queries & keys ($\theta = 10,000$); zero trainable weights |
| **Feed-Forward Network** | SwiGLU ($d_{\text{ffn}} = 2048$) | Gated linear activation with SiLU providing superior non-linear representation |
| **Vocabulary & Head** | 32,768 Byte-level BPE | Tied input/output embeddings eliminating 25.17M redundant parameters |
| **Linear Biases** | Disabled globally | Modern LLaMA-style parameter efficiency and improved training stability |

---

## Lifecycle Benchmark Matrix

Every stage of NexoraLM was empirically validated, logged, and verified:

| Stage | Model Variant | Precision | Parameters (Trainable) | Weights Size | Throughput / Hardware | Empirical Result / Metrics |
|:---|:---|:---|:---|:---|:---|:---|
| **M1: Tokenizer** | Byte BPE | N/A | 32,768 Vocab | 2.3 MB | CPU | 7 Special tokens, `<|assistant|>` masking |
| **M2: Micro Sanity** | NexoraMicro | FP32 | 18.88M (100%) | 2.1 GB VRAM | GPU | Overfit loss: **0.0054**, Bitwise checkpoint restore |
| **M4: Hardware Gate**| NexoraLM-Gate | FP16/SDPA | 125.85M (100%) | 5.85 GB VRAM | 11,042.4 tok/s | Tesla T4 compute gate verified: **GO** |
| **M4+: Base Pretrain**| NexoraLM-Base | FP16/SDPA | 125.85M (100%) | 240.0 MB | Tesla T4 (8.5h) | **299.7M tokens seen**, Final loss: **3.8744** |
| **M5: SFT** | NexoraLM-SFT | FP16 | 125.85M (100%) | 240.0 MB | Local GPU | OASST1 tuning, Val Loss: **3.5177**, PPL: **33.71** |
| **M6: LoRA** | NexoraLM-LoRA | FP16+Adapters | 1,310,720 (1.03%) | ~5.2 MB | Local GPU | Targets `q, k, v, o` ($r=16, \alpha=32$) |
| **M7: QLoRA** | NexoraLM-QLoRA| 4-bit+Adapters| 1,310,720 (1.03%) | 64.7 MB Base | Local GPU | 4-bit NF4 quantized base, FP16 adapters |
| **M8: DPO** | NexoraLM-DPO | FP16 | 125.85M (Policy) | 240.0 MB | Local GPU | DPO loss: **0.0030**, Reward margin: **+5.8164** (100% Acc) |
| **M10: INT8 Quant** | NexoraLM-INT8 | INT8 | Static Quantized | 120.55 MB (50%) | CPU GEMM | Mean Relative Error: **0.0078** (< 0.8%) |
| **M10: INT4 Quant** | NexoraLM-INT4 | INT4 | Group-wise (g=128)| 64.69 MB (27%) | 3.7x compression | Mean Relative Error: **0.1006** (~10.0%) |
| **M11: Serving** | FastAPI SSE | FP16 / INT8 | 125.85M | Production API | SSE Stream | TTFT: **34.2 ms**, OpenAI format compliant |

---

## Repository Layout

```text
NexoraLM/
├── website/            # Premium web app (Interactive console, live SSE, metrics)
│   ├── index.html      # Responsive glassmorphism interface
│   ├── style.css       # Custom design system with modern typography & tokens
│   └── app.js          # Live FastAPI SSE integration with dynamic fallback
├── src/
│   ├── tokenizer/      # Byte-level BPE tokenizer (32,768 vocab, assistant masking)
│   ├── model/          # 126M Decoder-only Transformer & tied embeddings
│   ├── attention/      # Grouped-Query Attention (GQA) & iterative KV-cache
│   ├── rope/           # Rotary Position Embeddings (RoPE)
│   ├── normalization/  # Root Mean Square Layer Normalization (RMSNorm)
│   ├── ffn/            # SwiGLU feed-forward network
│   ├── training/       # AdamW, cosine schedule, FP16 GradScaler, checkpoints
│   ├── lora/           # Custom LoRA implementation & weight merging
│   ├── dpo/            # Custom DPO loss and implicit preference reward margins
│   ├── quantization/   # INT8 per-channel & INT4 group-wise quantization
│   ├── inference/      # Autoregressive generation (greedy, top-p, top-k, temp)
│   └── serving/        # FastAPI inference server with SSE streaming & CORS
├── configs/            # Parameterized YAML configs for all training phases
├── scripts/            # Pipeline runners (pretrain, sft, lora, qlora, dpo, quantize, serve)
├── tests/              # 22 automated unit tests covering math, shapes, and serving
├── docs/               # 14 complete engineering architecture & review dossiers
└── outputs/            # Empirical evaluation logs and benchmark reports
```

---

## Quickstart Guide

### 1. Environment Setup
```bash
git clone https://github.com/Ravikishore710/NexoraLM.git
cd NexoraLM
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .
```

### 2. Run Automated Test Suite
```bash
pytest tests/ -v
```
All 22 unit tests validate causal attention gradient isolation, KV cache numerical equivalence, parameter counts, LoRA weight merging, DPO loss, and INT8/INT4 quantization fidelity.

### 3. Launch Local API Server
```bash
python scripts/serve.py --checkpoint checkpoints/nexoralm_aligned_300m.pt --port 8000
```

### 4. Interactive Terminal Chat
```bash
python scripts/chat_cli.py --url http://localhost:8000
```

### 5. Launch Showcase Web Console
```bash
python -m http.server 3000 --directory website
```
Navigate to `http://localhost:3000` to interact with the model via the web interface.

---

## Technical Highlights

- **Pure First-Principles Implementation**: No Hugging Face Transformer black-box imports for the core model; written directly in PyTorch tensor algebra.
- **True KV-Cache Streaming**: Autoregressive decoding reuses key and value projections with 3:1 GQA head grouping, eliminating duplicate prompt computations.
- **Full Alignment Lifecycle**: Base pretraining on ~300M tokens from FineWeb-Edu, followed by SFT on instruction pairs and closed-form DPO preference optimization.
- **Production Efficiency**: Per-channel INT8 and group-wise INT4 quantization providing up to 3.7x weight compression for lightweight edge deployment.

---

## Author

**Ravi Kishore**  
GitHub: [@Ravikishore710](https://github.com/Ravikishore710)  
Repository: [NexoraLM](https://github.com/Ravikishore710/NexoraLM)
