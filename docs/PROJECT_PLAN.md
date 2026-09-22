# NexoraLM Project Plan

## Overview
NexoraLM is a compact decoder-only language model (~126M parameters) built from first principles and taken through the complete model lifecycle:
Tokenizer -> Architecture -> Micro Validation -> Pretraining -> SFT -> LoRA -> QLoRA -> DPO -> Quantization -> KV-Cache Inference -> API Serving.

## Compute Positioning & Constraint
- Hardware Target: NVIDIA T4 (16 GB VRAM) on Kaggle.
- Pretraining Budget: 300M tokens on FineWeb-Edu (~2.38 tokens/parameter).
- Sanity Micro-Model: ~20M parameters on TinyStories.
- Framing: Proof-of-lifecycle decoder-only language model demonstrating full systems and engineering depth under concrete resource constraints.

## Frozen Clarifications & Invariants
1. M2 TinyStories separation: Small deterministic train split subset for training; validation split for evaluation only.
2. Total vocabulary: Exactly 32,768 tokens, including all base byte tokens and special tokens (<BOS>, <EOS>, <PAD>, <UNK>, <|system|>, <|user|>, <|assistant|>).
3. Tokenizer data pool: Sampled strictly from the FineWeb-Edu training split pool.
4. DPO reference model: NexoraLM-SFT is cloned into policy and frozen reference.
5. LoRA vs QLoRA: LoRA uses FP16 base; QLoRA uses 4-bit base weights.
6. Quantization scheme: INT8 per-channel symmetric; INT4 group-wise (group size 128) with scale and zero-point.
7. Compute gate: 500-2,000 step throughput test on Kaggle T4 before committing the 300M token pretraining run.

## Milestone Roadmap
- M0: Skeleton, configuration, test harness, 14 documentation files.
- M1: Byte-level BPE tokenizer (32,768 vocabulary).
- M2: Micro-model (~20M) validation on TinyStories train split.
- M3: 126M architecture from first principles.
- M4: Pretraining data pipeline, throughput gate, 300M token run.
- M5: SFT with assistant-only loss masking.
- M6: LoRA implementation and benchmark against Full SFT.
- M7: QLoRA implementation and benchmark.
- M8: DPO alignment with frozen SFT reference.
- M9: KV-cache autoregressive inference.
- M10: Quantization benchmarks (FP16, INT8, INT4).
- M11: FastAPI serving with SSE streaming.
- M12: Evaluation matrix and ablations.
- M13: 40-topic technical interview dossier.
- M14: Final freeze, README, reproducibility verification.
