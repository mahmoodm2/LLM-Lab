# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**LLM Fine-Tuning & Reasoning Lab** — a hands-on learning environment for building production-grade LLM systems on Google Colab Pro.

- **Base models**: Qwen2.5-1.5B (T4) / Qwen2.5-7B or Qwen3 (A100)
- **Core stack**: Unsloth + TRL + Hugging Face ecosystem
- **Reference repos** (read-only): `rasbt/LLMs-from-scratch` (Parts 0–2), `rasbt/reasoning-from-scratch` (Part 5)
- **Your codebase**: `llm-finetuning-lab` (single repo, all phases)

All conceptual background lives in `CONCEPTS.md`.

## Planned Structure

```
llm-finetuning-lab/
├── notebooks/
│   ├── 00_data_sft.ipynb               ← Part 1: SFT dataset from domain text
│   ├── 00_data_preference.ipynb        ← Part 1: DPO preference pairs
│   ├── 01_sft_qwen.ipynb               ← Part 2: QLoRA SFT with Unsloth + TRL
│   ├── 02_dpo.ipynb                    ← Part 3: DPO alignment
│   ├── 03_grpo.ipynb                   ← Part 3: GRPO with rule-based rewards
│   ├── 04_inference.ipynb              ← Part 4: vLLM, batching, prefix cache
│   ├── 05_speculative_decoding.ipynb   ← Part 4: draft model benchmarking
│   ├── 06_quantization.ipynb           ← Part 4: AWQ vs GGUF tradeoffs
│   ├── 07_reasoning_distillation.ipynb ← Part 5: CoT distillation from teacher
│   └── 08_reasoning_grpo.ipynb         ← Part 5: GRPO for reasoning
├── src/
│   ├── data/
│   │   ├── chunk.py        ← split raw docs into 512–1024 token passages
│   │   ├── synthesize.py   ← call LLM API to generate instruction/response pairs
│   │   ├── filter.py       ← length, dedup, perplexity, format filtering
│   │   ├── format.py       ← apply chat template and tokenize
│   │   └── preference.py  ← build chosen/rejected pairs for DPO
│   ├── train.py            ← shared training entry point
│   └── eval.py             ← evaluation helpers
└── configs/
    ├── sft_config.yaml
    ├── dpo_config.yaml
    └── grpo_config.yaml
```

## Architecture & Phase Dependencies

Phases are sequential; each checkpoint feeds downstream:

```
Part 0 (Foundations) → informs all parts

Part 1 (Data) → feeds Part 2 (SFT dataset), Part 3 (preference/verifiable data), Part 5 (CoT traces)

Part 2 (SFT) → checkpoint feeds Part 3 (both DPO and GRPO start from SFT)
             → checkpoint feeds Part 4 (inference experiments on trained model)

Part 3 (Alignment) → aligned checkpoint feeds Part 5

Part 4 (Inference) → independent; can run in parallel with Part 3/5

Part 5 (Reasoning) → extends Part 3 GRPO with reasoning-specific reward design
```

**SFT is the convergence point** — all alignment and reasoning work branches from the SFT checkpoint.

## Hardware Constraints

Every notebook must run on T4 + Qwen2.5-1.5B. A100 experiments are optional scale-ups.

| GPU | VRAM | Model fit |
|-----|------|-----------|
| T4  | 16GB | 1.5B full; 7B QLoRA |
| A100 | 40GB | 7B full; 13B QLoRA |

Default training setup: QLoRA (4-bit NF4) + Unsloth (2x memory savings on top of QLoRA). Always enable gradient checkpointing on T4.

## Key Patterns

**Model loading** (Unsloth): `FastLanguageModel.from_pretrained()` + `get_peft_model()`. Unsloth patches transparently — TRL trainers don't need changes.

**Chat template**: Qwen has its own template. During SFT, prompt tokens must be masked from loss (labels = -100). Same template at inference but prompt-only. Misalignment here cascades through all downstream phases.

**Checkpoint strategy**: Save both merged weights (`base + adapter`) and unmerged LoRA adapters. Downstream phases load the merged checkpoint; unmerged adapters allow inference flexibility.

**GRPO reward design**:
- Accuracy reward (exact match or execution-verified) = primary signal
- Format reward (`<think>...</think>` tags) = stability guard
- Detect reward hacking early: check if model learns spurious tricks rather than reasoning

**Synthetic data for reasoning**: Use teacher model (Claude, Qwen3, DeepSeek-R1) to generate `<think>...</think>` traces. Include "journey learning" traces with wrong paths and self-correction, not just perfect solutions.

## Trainer Defaults

| Trainer | lr | Key params |
|---------|-----|-----------|
| SFTTrainer | 2e-4 | batch=4–8, epochs=2–3, grad_accum=2–4, cosine LR with warmup |
| DPOTrainer | 5e-6 | beta=0.1–0.5 (KL penalty), epochs=1–2 |
| GRPOTrainer | 1e-5 | mini_batch=8, num_generation_steps=3–5 |