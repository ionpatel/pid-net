# PID-Net: Control-Theoretic Neural Architecture

## Core Idea

Replace standard linear transformations with PID (Proportional-Integral-Derivative) layers where each weight has three learnable components that capture different temporal dynamics of the signal, plus a gating mechanism that learns which components to activate.

## Mathematical Formulation

### Standard Linear Layer
```
y = Wx + b
```
This only sees the **current** input. No memory, no dynamics.

### PID Layer

Given input signal x(t) at timestep t:

```
y(t) = W_p · x(t) + W_i · S(t) + W_d · Δ(t) + b
```

Where:
- **W_p** (Proportional) — reacts to current input magnitude
- **W_i** (Integral) — reacts to accumulated input history
- **W_d** (Derivative) — reacts to rate of change

**Integral term (accumulated signal):**
```
S(t) = α · S(t-1) + (1-α) · x(t)
```
Exponential moving average with learnable decay α ∈ (0,1).
This is a soft integral — avoids unbounded growth while capturing history.

**Derivative term (rate of change):**
```
Δ(t) = x(t) - x(t-1)
```
Or smoothed: `Δ(t) = β · Δ(t-1) + (1-β) · (x(t) - x(t-1))`

### Adaptive Gating (the key innovation)

The model **learns when to use each component:**

```
g(t) = σ(W_gate · [x(t); S(t); Δ(t)] + b_gate)    # ∈ R^3, softmax or sigmoid

y(t) = g_p(t) · (W_p · x(t)) + g_i(t) · (W_i · S(t)) + g_d(t) · (W_d · Δ(t)) + b
```

Where g(t) = [g_p, g_i, g_d] are per-timestep gate values.

**Why this matters:**
- For **stable signals** → model learns to emphasize P (proportional)
- For **trending signals** → model learns to emphasize I (integral catches drift)
- For **volatile signals** → model learns to emphasize D (derivative catches spikes)
- The model **adapts per-token, per-layer, per-channel**

## Architecture Variants

### Variant 1: PID-Linear (drop-in replacement)

Replace any nn.Linear with PIDLinear. Works for sequences.

```
Input: x ∈ R^(B × T × d_in)
Output: y ∈ R^(B × T × d_out)

Parameters:
  W_p ∈ R^(d_out × d_in)    — proportional weights
  W_i ∈ R^(d_out × d_in)    — integral weights  
  W_d ∈ R^(d_out × d_in)    — derivative weights
  α   ∈ R^(d_in)            — integral decay (learnable, sigmoid-constrained)
  β   ∈ R^(d_in)            — derivative smoothing (learnable)
  W_gate ∈ R^(3 × 3·d_in)  — gating network
```

### Variant 2: PID-Attention

Replace Q/K/V projections with PID projections:

```
Q(t) = W_p^q · x(t) + W_i^q · S(t) + W_d^q · Δ(t)
K(t) = W_p^k · x(t) + W_i^k · S(t) + W_d^k · Δ(t)
V(t) = W_p^v · x(t) + W_i^v · S(t) + W_d^v · Δ(t)
```

This gives attention **awareness of signal dynamics** — queries can attend based on trends and momentum, not just current values.

### Variant 3: PID-SSM (State Space Model hybrid)

Combine with Mamba-style selective state spaces:

```
h(t) = A · h(t-1) + B_pid · [x(t); S(t); Δ(t)]
y(t) = C · h(t) + D_pid · [x(t); S(t); Δ(t)]
```

The PID components feed into the state space, giving it richer input dynamics.

## Efficiency: Adaptive Computation

The gate mechanism enables **dynamic sparsity:**

```python
# If gate value < threshold, skip that component entirely
if g_p < ε:
    skip proportional computation  # save one matmul
if g_i < ε:
    skip integral computation      # save one matmul  
if g_d < ε:
    skip derivative computation    # save one matmul
```

**In practice:**
- Early layers might only need P (current features)
- Middle layers might use P+I (features + context)
- Final layers might use all three
- The model learns this automatically via the gates

This gives you **1x to 3x compute range per layer** — the model decides.

### Sparsity Regularization

Add a loss term to encourage component dropping:

```
L_sparse = λ · Σ_t Σ_layers entropy(g(t))
```

Low entropy = confident component selection = faster inference.

## Why This Could Work

### Theoretical Grounding

1. **PID controllers are universal** — they can approximate any continuous controller. By making the weights PID-structured, each neuron becomes a tiny controller.

2. **Integral = implicit memory** — the I component gives every layer temporal memory without explicit recurrence. This is cheaper than attention over the full sequence.

3. **Derivative = change detection** — the D component makes the network naturally sensitive to transitions, edges, and anomalies.

4. **Gating = mixture of experts over time dynamics** — the model learns which temporal scale matters for each feature.

### Comparison to Existing Approaches

| Approach | Handles History | Handles Change | Adaptive | Extra Params |
|----------|----------------|----------------|----------|-------------|
| Linear | ❌ | ❌ | ❌ | 0 |
| LSTM | ✅ (cell state) | ❌ (implicit) | ❌ | 4x |
| Transformer | ✅ (attention) | ❌ (implicit) | ❌ | 3x (QKV) |
| Mamba/SSM | ✅ (state) | ❌ (implicit) | ✅ | 2x |
| **PID-Net** | ✅ (integral) | ✅ (derivative) | ✅ (gates) | 3x + gate |

Key advantage: **explicit** handling of both history AND rate-of-change, with learned gating. No one else has this.

## Target Applications

1. **Time Series Forecasting** — natural fit, PID captures trends + momentum + level
2. **Trading** — price, volume, and order flow are inherently PID signals
3. **Language** — word frequency (I), topic shifts (D), current token (P)
4. **Control Systems** — obviously
5. **Audio/Speech** — pitch (P), melody (I), transients (D)

## Experiments to Run

### Experiment 1: Synthetic Control Task
- Generate data from actual PID controllers with varying P/I/D gains
- Train PID-Net vs MLP vs LSTM to predict controller output
- Verify PID-Net recovers the true P/I/D structure

### Experiment 2: Time Series Benchmarks
- ETTh1/ETTh2, Weather, Electricity datasets
- Compare: PID-Linear vs Linear vs Transformer vs Mamba
- Measure accuracy AND FLOPs

### Experiment 3: Trading Signal Prediction
- Use our own market data (SOL, DOGE, ETH)
- Predict next-candle direction and magnitude
- Compare PID-attention vs standard attention

### Experiment 4: Language Modeling (stretch)
- Small-scale LM (125M params)
- Replace all Linear layers with PID-Linear
- Compare perplexity and training dynamics

## Open Questions

- [ ] Best initialization for W_i, W_d, α, β?
- [ ] Should α/β be per-channel, per-layer, or global?
- [ ] Can we make the integral truly continuous (Neural ODE style)?
- [ ] How does PID-Net interact with normalization (LayerNorm, RMSNorm)?
- [ ] Gradient flow through the integral/derivative — any vanishing/exploding issues?
- [ ] Can we distill a trained PID-Net into a simpler model by reading the gates?

## Related Work

- **Neural ODEs** (Chen et al., 2018) — continuous-time dynamics, but no explicit PID structure
- **PID optimizers** — use PID for learning rate control, not as architecture
- **Liquid Neural Networks** (Hasani et al., 2021) — continuous-time RNNs, related but different
- **Mamba** (Gu & Dao, 2023) — selective state spaces, similar efficiency motivation
- **Adaptive Computation** (Graves, 2016) — dynamic compute per step

---

*Status: Research spec — ready for implementation*
*Last updated: 2026-03-12*
