# PID-Net 🧠

**PID Control Theory as Neural Architecture**

A novel neural network architecture where every linear transformation decomposes into three temporal streams:
- **P (Proportional)** — reacts to current input (perception)
- **I (Integral)** — accumulates past context (memory/experience)
- **D (Derivative)** — detects rate of change (intuition/anticipation)

An adaptive **gate** learns which stream to trust for each input — acting as a form of proto-consciousness.

## Why?

Standard neural networks are stateless at the layer level. They process what's in front of them with no sense of temporal dynamics. PID-Net changes this:

| Component | What it does | Cognitive parallel |
|-----------|-------------|-------------------|
| P (Proportional) | Processes current input | Perception |
| I (Integral) | Carries accumulated history | Memory |
| D (Derivative) | Detects what's changing | Intuition |
| Gate | Decides what matters now | Attention/consciousness |

## Results

### Experiment 1: Synthetic PID Recovery
| Model | Best MSE | vs PID-Net |
|-------|---------|-----------|
| **PID-Net** | **0.0018** | — |
| LSTM | 0.013 | 7.1x worse |
| MLP | 0.444 | 242x worse |

PID-Net reached target accuracy at epoch 65. Neither baseline reached it in 200 epochs.

### Experiment 2: Language Modeling (coming soon)
PID-Transformer vs Standard Transformer on Shakespeare character-level LM.

## Architecture

```
Input → [P: current] [I: cumulative] [D: Δchange]
                    ↓
            Learned Gate (softmax)
                    ↓
        Weighted combination → Output
```

### PID-Transformer
Full transformer with PID-aware Q/K/V projections:
- Queries carry temporal context
- Keys encode what + history + change
- Values are temporally enriched
- ~7.2M params (fits on Apple M2 with 8GB)

## Quick Start

```bash
# Install
pip install torch

# Run synthetic experiment
python exp1_minimal.py

# Train PID-Transformer on Shakespeare (auto-downloads data)
python train.py --compare --epochs 20 --generate

# Apple Silicon will auto-detect MPS backend
```

### Training Options
```bash
python train.py --epochs 20              # PID-Transformer only
python train.py --baseline --epochs 20   # Standard Transformer only  
python train.py --compare --epochs 20    # Both (head-to-head)
python train.py --compare --generate     # Both + text generation demo

# Smaller config for constrained hardware
python train.py --compare --d-model 128 --n-layers 4 --batch-size 16
```

## Files

| File | Description |
|------|-------------|
| `pid_layer.py` | Core PIDLinear layer + PIDBlock + PIDNet |
| `pid_attention.py` | PID-Transformer + Standard Transformer baseline |
| `train.py` | Training pipeline (MPS/CUDA/CPU, Shakespeare LM) |
| `exp1_minimal.py` | Experiment 1: synthetic PID recovery |
| `RESEARCH.md` | Full mathematical formulation |
| `VISION.md` | Cognitive architecture vision |

## The Vision

> The soul isn't a prompt. It's the learned gate biases.

After training, the gate develops a default mode — a cognitive style that emerges from data, not instructions. This is the path from language model to mind.

## License

MIT
