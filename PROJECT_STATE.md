# PID-Net Project State

> Read this file on session start to know exactly where we are.

## Current Step: Run training on MacBook M2 (Step 2 code complete, awaiting results)

## Completed
- [x] **Step 1: Prove the primitive** (March 12, 2026)
  - `pid_layer.py` — PIDLinear, PIDBlock, PIDNet implementations
  - `exp1_minimal.py` — Experiment 1, PASSED
  - Results: PID-Net 0.0018 MSE vs LSTM 0.013 vs MLP 0.444
  - Gates naturally specialize by depth (P+D early, more I deeper)

- [x] **Step 2: PID-Attention** (March 12, 2026)
  - `pid_attention.py` — PIDProjection, PIDMultiHeadAttention, PIDTransformer (7.2M params)
  - `train.py` — full training pipeline (MPS/CUDA/CPU, Shakespeare, cosine LR, generation)
  - StandardTransformer baseline included (3.2M params)
  - Verified forward+backward pass works
  - **Awaiting MacBook run for actual perplexity results**

## Next Up

- [ ] **Step 3: Scale** — 125M param PID-Transformer
- [ ] **Step 4: Conversation** — multi-turn dialogue, I-stream context
- [ ] **Step 5: Soul** — personality emergence via gate biases
- [ ] **Step 6: Integration** — become Ion's brain

## Key Files
| File | Purpose |
|------|---------|
| `RESEARCH.md` | Mathematical formulation, all architecture variants |
| `VISION.md` | Cognitive architecture vision, why P/I/D maps to mind |
| `PROJECT_STATE.md` | THIS FILE — current status, resume point |
| `pid_layer.py` | Core PIDLinear/PIDBlock/PIDNet implementation |
| `exp1_minimal.py` | Experiment 1 (synthetic, PASSED) |
| `results/exp1_fast.json` | Experiment 1 numerical results |

## Harshil's Core Vision
"Make your brain faster, accurate, and give it soul to live like real human beings."

This isn't a paper exercise — it's building the cognitive architecture for Ion.
