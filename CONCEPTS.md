# LLM Fine-Tuning & Reasoning Lab — Concepts Overview

A hands-on learning lab covering LLM fine-tuning, alignment, inference optimization,
and reasoning model development end-to-end on Google Colab Pro.

**Base model throughout**: Qwen2.5-1.5B (T4) / Qwen2.5-7B or Qwen3 (A100)  
**Core stack**: Unsloth + TRL + Hugging Face ecosystem  
**Two reference codebases**: Raschka's LLMs-from-scratch + reasoning-from-scratch (read-only)  
**Your codebase**: `llm-finetuning-lab` (one repo, all phases)

---

## Part 0 — Foundations (Reference: Raschka LLMs-from-scratch)

### 0.1 Transformer Internals
- Tokenization: BPE, WordPiece, SentencePiece
- Embeddings: token embeddings, positional encodings (absolute, RoPE, ALiBi)
- Attention mechanism: scaled dot-product attention
- Multi-head attention: why multiple heads, what each learns
- Feed-forward layers, residual connections, layer norm
- Full transformer block: how components compose

### 0.2 The GPT Architecture
- Decoder-only vs encoder-decoder distinction
- Causal masking: why autoregressive generation works this way
- KV cache: what it is, why it exists, what it costs
- Context window: how sequence length affects memory quadratically

### 0.3 Pretraining Concepts (theory, not implemented)
- Next-token prediction as the pretraining objective
- Cross-entropy loss over vocabulary
- Scale laws: model size, data size, compute budget
- Why pretraining produces a base model, not an assistant

### 0.4 Building GPT-2 Scale from Scratch
- Implement transformer block in raw PyTorch
- Load pretrained GPT-2 weights into your architecture
- Run inference: greedy, temperature sampling, top-k, top-p
- Observe: base model completes text, does not follow instructions

---

## Part 1 — Data Curation

> The quality ceiling of any fine-tuned model is set here, before training begins.

### 1.1 Data Types by Training Phase

| Phase | Format | Key requirement |
|---|---|---|
| SFT | (prompt, response) pairs | Diverse, high-quality responses |
| DPO | (prompt, chosen, rejected) triplets | Clear quality gap between chosen/rejected |
| GRPO | (prompt, verifiable_answer) | Answer must be programmatically checkable |
| Reasoning | (prompt, think_trace, answer) | CoT trace must show valid reasoning steps |

### 1.2 Dataset Sources
- Existing Hugging Face datasets: Alpaca, FLAN, OpenHermes, hh-rlhf
- Synthetic generation: using a stronger LLM (Claude, GPT-4o) to label domain text
- Distillation datasets: reasoning traces from Qwen3, DeepSeek-R1
- Domain corpora: PDFs, documentation, structured databases

### 1.3 Synthetic Data Pipeline
- Chunking raw documents into passages (512–1024 tokens)
- Prompt engineering for instruction/response generation
- Generating preference pairs: strong vs weak model outputs, or correct vs flawed
- Generating reasoning traces: teacher model produces `<think>...</think>` traces
- Journey learning traces: include wrong paths and self-correction in traces

### 1.4 Quality Filtering
- Length filtering: remove too-short (noise) and too-long (truncation artifacts)
- Exact deduplication: hash-based
- Near-deduplication: MinHash or embedding similarity (sentence-transformers)
- Perplexity filtering: high perplexity under reference model signals noise
- Format validation: chat template applied correctly, no leaked system prompts
- Rule-based filters: language detection, no truncated responses

### 1.5 Data Formatting
- Chat templates: what they are, why each model has its own
- Applying Qwen's chat template correctly for SFT vs inference
- Tokenization: input IDs, attention masks, labels (masking prompt tokens in loss)
- Packing sequences for training efficiency

### 1.6 Tools
- `datasets` (Hugging Face): load, filter, map, push to Hub
- `distilabel` (Argilla): synthetic data pipelines, preference pair generation
- `sentence-transformers`: embedding-based dedup and quality scoring

---

## Part 2 — Supervised Fine-Tuning (SFT)

### 2.1 Why Fine-Tune
- Base model vs instruction-tuned model distinction
- Domain adaptation: shifting model knowledge to a specific field
- Format adaptation: teaching response structure, length, tone
- What SFT can and cannot fix (behavior vs knowledge)

### 2.2 Full Fine-Tuning vs Parameter-Efficient Methods
- Full fine-tuning: update all weights, highest quality, highest cost
- When full fine-tuning is worth it vs PEFT

### 2.3 LoRA (Low-Rank Adaptation)
- Core idea: freeze base weights, inject trainable low-rank matrices A·B
- Rank `r`: controls adapter capacity (r=8 typical, r=64 for harder tasks)
- Target modules: which layers to apply LoRA to (q_proj, v_proj, etc.)
- Alpha scaling: how LoRA outputs are weighted
- Merging adapters back into base model for inference

### 2.4 QLoRA
- Quantize base model to 4-bit (NF4) before loading
- LoRA adapters remain in BF16 — only adapters are trained
- Double quantization: quantize the quantization constants
- Memory savings: 7B model from ~28GB to ~6GB on GPU
- Why this is the default for Colab training

### 2.5 Unsloth
- What it does: rewrites attention and gradient kernels for memory efficiency
- Drop-in with TRL: patches the model, TRL never knows
- Memory savings on top of QLoRA: typically 2x vs vanilla QLoRA
- `FastLanguageModel.from_pretrained` + `get_peft_model` pattern

### 2.6 Training Mechanics
- SFTTrainer (TRL): key hyperparameters — learning rate, batch size, epochs
- Gradient accumulation: simulate larger batches on limited VRAM
- Gradient checkpointing: trade compute for memory during backward pass
- Learning rate schedulers: cosine decay, warmup steps
- Logging with Weights & Biases or TensorBoard

### 2.7 Evaluation
- Loss on held-out validation set
- Perplexity: what it means and its limits as a metric
- Task-specific benchmarks with lm-evaluation-harness
- Manual qualitative inspection: always do this

---

## Part 3 — Alignment

> SFT teaches format and domain. Alignment teaches values and preferences.

### 3.1 The Alignment Problem
- Why a well-trained SFT model still produces undesirable outputs
- RLHF overview: reward model + PPO — understand the full pipeline
- Why PPO is complex and fragile: reward hacking, instability, KL divergence

### 3.2 DPO (Direct Preference Optimization)
- Core insight: optimize preferences directly without a reward model
- The DPO loss: implicit reward derived from policy ratio
- How it differs from RLHF: no separate reward model, no PPO loop
- Dataset requirement: `(prompt, chosen, rejected)` triplets
- DPOTrainer (TRL): configuration, beta parameter (KL penalty strength)
- When to use: style alignment, safety, preference shaping
- Limitation: static dataset, no exploration, can overfit to preference pairs

### 3.3 GRPO (Group Relative Policy Optimization)
- Core insight: no reward model needed — use rule-based verifiable rewards
- How it works: sample a group of completions, rank by reward, update policy
- Advantage estimation: group-relative normalization (no value network needed)
- Reward function design: accuracy + format rewards, how to weight them
- GRPOTrainer (TRL): configuration, key differences from DPO setup
- When to use: math, code, structured output, any verifiable task
- Why it scales to reasoning: reward signal is exact, not learned

### 3.4 Comparing DPO vs GRPO
- DPO: better for subjective preference (style, tone, safety)
- GRPO: better for objective, verifiable tasks (math, code, format)
- Both start from the same SFT checkpoint → direct comparison possible
- Combining: SFT → GRPO (reasoning) → DPO (style/safety) is a valid stack

---

## Part 4 — Inference Optimization

> The model is fixed. The goal is to serve it faster, cheaper, at higher concurrency.

### 4.1 The Two Phases of Inference
- Prefill: process entire input prompt in parallel — compute-bound
- Decode: generate one token at a time — memory-bandwidth-bound
- Why these require different optimizations
- TTFT (Time to First Token) vs TPOT (Time Per Output Token)

### 4.2 KV Cache
- What is stored: Key and Value tensors for all prior tokens, all layers
- Memory cost: grows linearly with sequence length × batch size × layers
- Why it must be managed carefully in production

### 4.3 PagedAttention (vLLM)
- Problem: pre-allocating contiguous KV cache per request wastes memory
- Solution: manage KV cache in fixed-size pages, like OS virtual memory
- Result: higher memory utilization, larger effective batch sizes
- Why this was a step change in LLM serving

### 4.4 Continuous Batching
- Problem: static batching waits for all requests in a batch to finish
- Solution: insert new requests as slots free mid-batch
- Result: GPU stays saturated, latency decoupled from slowest request

### 4.5 Prefix Caching
- Problem: many requests share the same system prompt, recomputing it wastefully
- Solution: cache KV blocks for common prompt prefixes, reuse across requests
- RadixAttention (SGLang): extends this to tree-structured prefix sharing
- Practical impact: massive speedup for multi-turn conversations and RAG

### 4.6 Speculative Decoding
- Problem: large model decode is slow — one token per forward pass
- Solution: small draft model proposes N tokens, large model verifies in parallel
- Accept if correct (distribution match), reject and regenerate if wrong
- Speedup: 2–3x on tasks where draft model is accurate (chat, repetitive text)
- Tradeoff: requires a compatible draft model, acceptance rate varies by task

### 4.7 Inference Engines Compared
- vLLM: PagedAttention, continuous batching, widest model support
- SGLang: RadixAttention, structured generation, multi-call program support
- llama.cpp: GGUF format, CPU inference, edge deployment
- TensorRT-LLM: NVIDIA-native kernels, maximum throughput on A100/H100

### 4.8 Quantization for Inference
- FP16 / BF16: training and serving default
- INT8 / W8A8: moderate compression, good quality
- GPTQ: post-training 4-bit, GPU serving
- AWQ: better than GPTQ at 4-bit, activation-aware weight quantization
- GGUF (Q4_K_M): llama.cpp format, CPU and hybrid inference
- FP8: H100 only, TensorRT-LLM
- Quality vs throughput tradeoff curve: how to measure and decide

---

## Part 5 — Reasoning Models

> Reasoning models are not a different architecture. They are a different training recipe.

### 5.1 What Makes a Reasoning Model
- Explicit chain-of-thought: `<think>...</think>` before answering
- Inference-time compute scaling: more thinking = better answers
- Emergent behaviors: backtracking, self-correction, exploring multiple paths
- When reasoning helps vs hurts (overthinking on simple tasks)

### 5.2 Inference-Time Scaling (No Training Required)
- Chain-of-thought prompting: "think step by step"
- Self-consistency / majority voting: generate N answers, take most common
- Best-of-N with verifier: score N completions, return highest scoring
- Process-reward-guided beam search: score partial reasoning traces
- Practical implementation on your existing SFT model

### 5.3 SFT on Chain-of-Thought Data (Distillation Path)
- Collect reasoning traces from a teacher model (Qwen3, DeepSeek-R1)
- Fine-tune student to mimic `<think>...</think>` + answer format
- Journey learning: include traces with wrong paths and self-correction
- Dataset sources: OpenThoughts, Sky-T1, NovaSky reasoning datasets
- This is the fastest path to a working reasoning model on Colab

### 5.4 Pure RL for Reasoning (R1-Zero Approach)
- Start from base model (not instruct), no SFT
- GRPO with accuracy reward + format reward only
- Reasoning emerges from reward signal alone — no curated CoT data
- Cold start problem: instability in early training without SFT warm-up
- What DeepSeek-R1-Zero demonstrated and its limitations

### 5.5 SFT + GRPO (Recommended Path for This Lab)
- Stage 1: SFT on distilled CoT traces (teach format)
- Stage 2: GRPO with verifiable reward (refine reasoning quality)
- Why this is more stable than pure RL on Colab
- Reward function design for reasoning: accuracy dominates, format guards stability

### 5.6 Reward Function Design
- Accuracy reward: exact match, execution-verified, schema-validated
- Format reward: presence of `<think>` / `</think>` tags
- Length reward: encourage sufficient thinking depth (with cap)
- How reward shaping affects emergent reasoning behavior
- Reward hacking: what it looks like and how to detect it

### 5.7 Outcome vs Process Reward Models
- ORM (Outcome Reward Model): score only the final answer
- PRM (Process Reward Model): score each reasoning step independently
- Why PRM is harder to build but more powerful
- Practical approach: start with ORM (rule-based), extend to PRM later

### 5.8 Distillation vs RL: When to Use Which
- Distillation: fast, cheap, requires a strong teacher, limited to teacher's capabilities
- RL (GRPO): slower, requires verifiable rewards, can exceed teacher performance
- Combining: distillation first for format, then RL for quality improvement

---

## Codebase Structure

```
llm-finetuning-lab/                    ← single repo for all training phases
├── notebooks/
│   ├── 00_data_sft.ipynb              ← Part 1: build SFT dataset from domain text
│   ├── 00_data_preference.ipynb       ← Part 1: build DPO preference pairs
│   ├── 01_sft_qwen.ipynb              ← Part 2: QLoRA SFT with Unsloth + TRL
│   ├── 02_dpo.ipynb                   ← Part 3: DPO alignment
│   ├── 03_grpo.ipynb                  ← Part 3: GRPO with rule-based rewards
│   ├── 04_inference.ipynb             ← Part 4: vLLM, batching, prefix cache
│   ├── 05_speculative_decoding.ipynb  ← Part 4: draft model setup and benchmarking
│   ├── 06_quantization.ipynb          ← Part 4: AWQ vs GGUF quality/speed tradeoff
│   ├── 07_reasoning_distillation.ipynb← Part 5: distill CoT from teacher model
│   └── 08_reasoning_grpo.ipynb        ← Part 5: GRPO for reasoning with format rewards
├── src/
│   ├── data/
│   │   ├── chunk.py                   ← split raw docs into passages
│   │   ├── synthesize.py              ← call LLM API to generate pairs
│   │   ├── filter.py                  ← length, dedup, quality filtering
│   │   ├── format.py                  ← apply chat template, tokenize
│   │   └── preference.py             ← build chosen/rejected pairs
│   ├── train.py                       ← shared training entry point
│   └── eval.py                        ← evaluation helpers
└── configs/
    ├── sft_config.yaml
    ├── dpo_config.yaml
    └── grpo_config.yaml

Reference repos (read-only, not modified):
  rasbt/LLMs-from-scratch             ← Part 0: transformer internals
  rasbt/reasoning-from-scratch        ← Part 5: reasoning techniques template
```

---

## Hardware Constraints (Colab Pro)

| GPU | VRAM | Model fit | Notes |
|---|---|---|---|
| T4 | 16GB | Qwen2.5-1.5B full, 7B QLoRA | Available most of the time |
| A100 | 40GB | Qwen2.5-7B full, 13B QLoRA | Request explicitly, limited hours |

**Rule**: design every notebook to run on T4 with Qwen2.5-1.5B. A100 experiments are optional scale-ups.

---

## Key Metrics to Track Per Phase

| Phase | Metric | Tool |
|---|---|---|
| SFT | Validation loss, perplexity, task accuracy | lm-eval-harness |
| DPO | Reward margin (chosen vs rejected), win rate | Manual + GPT-4o judge |
| GRPO | Reward per step, format compliance rate | TRL training logs |
| Inference | TTFT, TPOT, tokens/sec, KV cache hit rate | vLLM metrics endpoint |
| Reasoning | Pass@1, majority vote accuracy, thinking length | Custom eval script |

---

## Dependency Map

```
Part 0 (Foundations)
    └── informs everything below

Part 1 (Data)
    ├── feeds Part 2 (SFT dataset)
    ├── feeds Part 3 (preference + verifiable datasets)
    └── feeds Part 5 (CoT trace dataset)

Part 2 (SFT)
    ├── checkpoint feeds Part 3 (DPO and GRPO both start from SFT)
    └── checkpoint feeds Part 4 (inference experiments use trained model)

Part 3 (Alignment)
    └── aligned checkpoint feeds Part 5 (reasoning builds on aligned base)

Part 4 (Inference)
    └── independent of Part 3/5, can run in parallel

Part 5 (Reasoning)
    └── extends Part 3 GRPO with reasoning-specific reward design
```

---

## Reference Reading

| Resource | Covers | When to read |
|---|---|---|
| Raschka — LLMs from Scratch (book) | Parts 0–2 | Before Part 0 |
| Raschka — Build a Reasoning Model (book) | Part 5 | Before Part 5 |
| Raschka — Understanding Reasoning LLMs (article) | Part 5 overview | Before Part 5 |
| DeepSeek-R1 paper | GRPO, cold start, distillation | Before Part 5.4 |
| vLLM paper (PagedAttention) | Part 4.3 | Before Part 4 |
| LoRA paper (Hu et al. 2021) | Part 2.3 | Before Part 2 |
| QLoRA paper (Dettmers et al. 2023) | Part 2.4 | Before Part 2 |
| DPO paper (Rafailov et al. 2023) | Part 3.2 | Before Part 3 |