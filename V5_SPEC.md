# PID-Net v5: Hypergraph Rewriting Architecture
## The Computational Universe of Intelligence

**Authors:** Ion Patel & Harshil Patel  
**Date:** March 14, 2026  
**Status:** Specification / Pre-implementation  
**Framework:** MLX (Apple Silicon native)  
**Hardware Path:** MacBook → Mac Studio → Mac Studio Cluster  
**Inspired by:** PID Control Theory × Wolfram Physics × Predictive Coding × Fast Weight Programmers

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Why This is Different](#2-why-this-is-different)
3. [Core Architecture](#3-core-architecture)
4. [Mathematical Formulation](#4-mathematical-formulation)
5. [The PID Rewriting Rule](#5-the-pid-rewriting-rule)
6. [Multi-Scale Fractal Structure](#6-multi-scale-fractal-structure)
7. [Implementation Roadmap](#7-implementation-roadmap)
8. [Expectations & Milestones](#8-expectations--milestones)
9. [Testing & Debugging Framework](#9-testing--debugging-framework)
10. [Future Expansion](#10-future-expansion)
11. [Risk Analysis](#11-risk-analysis)
12. [Technical Challenges & Solutions](#12-technical-challenges--solutions)
13. [Appendix: Theoretical Foundations](#13-appendix-theoretical-foundations)

---

## 1. Executive Summary

**What:** A neural architecture where intelligence emerges from iterated application of learned rewriting rules on a dynamic hypergraph — not from stacking fixed layers.

**Why:** Every existing architecture (Transformers, Mamba, RWKV, Griffin) is a static function approximator: fixed computation graph, fixed operations per layer, fixed topology. They approximate intelligence; they don't grow it.

PID-Net v5 treats computation like physics: simple rules applied iteratively produce complex emergent behavior. The Wolfram Physics Project proved that spacetime, quantum mechanics, and general relativity all emerge from hypergraph rewriting. We apply the same principle to cognition.

**How:** Hidden states are nodes in a dynamic graph. Each processing step applies a **learned PID rewriting rule** that:
- **P (Perceive):** Read the current graph state via message passing
- **I (Integrate):** Store/retrieve past graph states via fast weight memory
- **D (Differentiate):** Compute prediction error between expected and actual graph evolution
- **Gate (Select):** Choose which rewriting rule to apply based on P/I/D signals

The same rule applied fractally at multiple scales (token → chunk → document) produces self-similar cognition — like Sierpiński triangles from a simple 1₃→3₃ rule.

**Competitive Edge:**
- Only architecture with explicit P, I, AND D streams (all others are P+I only)
- Only architecture where the computation graph itself evolves during inference
- Only architecture with fractal self-similarity by design
- Target complexity: O(T) per step via sparse graph operations

---

## 2. Why This is Different

### 2.1 The Paradigm Gap

| Paradigm | Examples | How it "thinks" |
|----------|----------|-----------------|
| Static layers | Transformer, MLP | Apply same function at each layer. Depth = capability. |
| Recurrent state | LSTM, Mamba, RWKV | Carry hidden state forward. State = memory. |
| Mixture of Experts | Switch, GLaM | Route to different experts. Sparsity = efficiency. |
| **Rewriting system** | **PID-Net v5** | **Evolve the computation graph itself. Emergence = intelligence.** |

Every architecture in the first three paradigms has a **fixed computation graph** determined at design time. The model learns weights within that graph, but the graph itself never changes. It's like trying to understand physics by picking the right equation — you'll get good approximations, but you'll never discover that spacetime itself is dynamic.

### 2.2 The Wolfram Insight

Wolfram's Physics Project demonstrated:
1. Simple rewriting rules on hypergraphs produce spacetime geometry
2. Different rules produce different physics (some produce spheres, some produce fractals)
3. The "right" rules produce our universe — general relativity and quantum mechanics as emergent properties

**Our claim:** The "right" learned rewriting rules on a cognitive hypergraph produce intelligence — perception, memory, prediction, and reasoning as emergent properties.

### 2.3 Why Not Just a GNN?

Graph Neural Networks (GNNs) operate on **fixed** graph topologies. They update node features via message passing, but the edges never change. PID-Net v5 differs in three critical ways:

1. **Dynamic topology:** Edges are created, strengthened, weakened, and removed during processing
2. **PID decomposition:** Every update decomposes into perceive/integrate/differentiate — not just "aggregate neighbor features"
3. **Multi-scale fractal:** The same rewriting rule operates at multiple levels of abstraction simultaneously

---

## 3. Core Architecture

### 3.1 Representation

**Cognitive Hypergraph** G = (V, E, H) where:
- **V** = {v₁, ..., vₙ} — nodes, each with feature vector vᵢ ∈ ℝᵈ
- **E** ⊆ V × V — directed edges with weights wᵢⱼ ∈ [0, 1]
- **H** — hyperedges connecting 3+ nodes (higher-order relationships)

For a sequence of T tokens:
- Each token becomes a node (or updates an existing node)
- Edges represent learned relationships between tokens
- Hyperedges capture multi-token patterns (phrases, clauses, concepts)

**State at step t:**
```
Gₜ = {
    nodes:      X ∈ ℝ^{N×d}          # node features
    edges:      A ∈ ℝ^{N×N}          # adjacency (soft, differentiable)
    hyperedges: H ∈ ℝ^{M×k×d}       # M hyperedges, each connecting k nodes
    memory:     W_fast ∈ ℝ^{d×d}     # fast weight matrix (I-stream)
    prediction: X̂ ∈ ℝ^{N×d}          # predicted next state (D-stream)
}
```

### 3.2 Processing Pipeline

For each new input token xₜ:

```
┌─────────────────────────────────────────────────────┐
│                   INPUT: token xₜ                    │
│                        │                             │
│                   ┌────▼────┐                        │
│                   │  EMBED  │  → node feature vₜ     │
│                   └────┬────┘                        │
│                        │                             │
│              ┌─────────▼──────────┐                  │
│              │   ADD TO GRAPH     │  → Gₜ gains node │
│              │   (initial edges   │                  │
│              │    from position)  │                  │
│              └─────────┬──────────┘                  │
│                        │                             │
│         ┌──────────────▼──────────────┐              │
│         │     PID REWRITING STEP      │ ← repeat R   │
│         │     (see Section 3.3)       │   times      │
│         └──────────────┬──────────────┘              │
│                        │                             │
│              ┌─────────▼──────────┐                  │
│              │     READOUT        │                   │
│              │  (predict next     │                   │
│              │   token from vₜ)   │                   │
│              └─────────┬──────────┘                  │
│                        │                             │
│                   OUTPUT: logits                      │
└─────────────────────────────────────────────────────┘
```

### 3.3 The PID Rewriting Step

This is the core operation. Applied R times per token (R can be adaptive).

```
┌───────────────────────────────────────────────────────────┐
│                 PID REWRITING STEP                         │
│                                                           │
│  Input: Gₜ (current graph)                                │
│                                                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │  P-STREAM    │  │  I-STREAM    │  │  D-STREAM    │    │
│  │  (Perceive)  │  │  (Integrate) │  │ (Differentiate)│  │
│  │              │  │              │  │              │    │
│  │ Message-pass │  │ Fast weight  │  │ Prediction   │    │
│  │ over current │  │ read/write   │  │ error:       │    │
│  │ graph edges  │  │ (memory of   │  │ ε = Gₜ - Ĝₜ │    │
│  │              │  │  past graphs)│  │              │    │
│  │ Xₚ = MP(G)  │  │ Xᵢ = W·q(X) │  │ Xd = f(ε)   │    │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘    │
│         │                 │                 │             │
│         └────────┬────────┴────────┬────────┘             │
│                  │                 │                       │
│           ┌──────▼──────┐  ┌──────▼──────┐                │
│           │    GATE     │  │   D→I       │                │
│           │  (select    │  │  COUPLING   │                │
│           │   blend of  │  │ (error      │                │
│           │   P, I, D)  │  │  drives     │                │
│           │             │  │  memory     │                │
│           │ gₚ,gᵢ,gd   │  │  write      │                │
│           │  = σ(...)   │  │  strength)  │                │
│           └──────┬──────┘  └──────┬──────┘                │
│                  │                │                        │
│           ┌──────▼────────────────▼──────┐                 │
│           │     APPLY REWRITE RULE       │                 │
│           │                              │                 │
│           │  X' = gₚ·Xₚ + gᵢ·Xᵢ + gd·Xd│                │
│           │  A' = update_edges(G, X')    │                 │
│           │  Ĝₜ₊₁ = predict(G')          │                │
│           │  W_fast += η·outer(k, v)     │                 │
│           └──────────────┬───────────────┘                 │
│                          │                                 │
│  Output: Gₜ' (rewritten graph)                             │
└───────────────────────────────────────────────────────────┘
```

### 3.4 Component Details

#### P-Stream: Graph Message Passing (Perception)

The P-stream reads the current graph state. Each node aggregates information from its neighbors:

```python
# Message passing (simplified)
messages = A @ X @ W_msg           # [N, d] — aggregate neighbor features
Xp = LayerNorm(X + messages)      # perceive current state
```

This is the "what's happening right now" signal. Unlike attention (which is quadratic), our message passing uses the learned sparse adjacency A, making it O(|E|) where |E| << N².

#### I-Stream: Fast Weight Programmer (Memory)

The I-stream maintains a **fast weight matrix** W_fast ∈ ℝᵈˣᵈ that serves as associative memory:

```python
# Write: store current graph state
key = W_key @ X_current            # what to index by
value = W_val @ X_current          # what to store
W_fast += η * outer(key, value)    # Hebbian write (outer product update)

# Read: retrieve relevant past context
query = W_query @ X_current
X_retrieved = W_fast @ query       # linear attention readout
Xi = LayerNorm(X_retrieved)
```

**Why this works:** Fast weight programming IS linear attention IS modern Hopfield networks IS associative memory. All mathematically equivalent (Schlag et al., 2021). The model writes current graph states into W_fast, then retrieves relevant past states via query. Unlike EMA (which is a blurry average), fast weights store DISCRETE, RETRIEVABLE memories.

**D→I Coupling:** The write strength η is modulated by the D-stream's prediction error. High error = surprising input = write more strongly to memory (like dopamine in the brain).

#### D-Stream: Prediction Error (Differentiation)

The D-stream implements **predictive coding** — the brain's actual algorithm:

```python
# At step t-1, predict what step t should look like
X_predicted = W_pred @ X_{t-1}     # prediction of next graph state

# At step t, compute prediction error
epsilon = X_actual - X_predicted    # what we got wrong
Xd = W_err @ epsilon               # project error into useful signal

# Update predictor (learn to predict better)
W_pred -= lr_pred * grad(||epsilon||²)
```

**Why prediction error, not finite difference:**
1. Finite difference (Xₜ - Xₜ₋₁) measures raw change — noisy, uninformative
2. Prediction error measures SURPRISE — how much reality deviated from expectations
3. The brain does predictive coding: top-down predictions, bottom-up errors
4. Only the error gets propagated — 10-100x more efficient than sending full states

#### Gate: Rewriting Rule Selection

The gate decides how to blend P/I/D and which rewriting variant to apply:

```python
# Compute gate values from all three streams
gate_input = concat(Xp.mean(), Xi.mean(), Xd.mean(), X.mean())
gate_logits = W_gate @ gate_input  # [3] for P/I/D blend
gp, gi, gd = softmax(gate_logits)

# Hierarchical: also decide whether to skip this rewrite entirely
skip_logit = W_skip @ gate_input   # scalar
skip = sigmoid(skip_logit)         # 0 = apply rewrite, 1 = skip (identity)

# Apply
X_new = skip * X + (1 - skip) * (gp * Xp + gi * Xi + gd * Xd)
```

**Adaptive compute:** When skip → 1, the layer does nothing (saves compute). When skip → 0 and gd is high, the model is surprised and doing heavy processing. This naturally allocates more compute to difficult/surprising parts of the input.

### 3.5 Edge Evolution (Dynamic Topology)

Unlike standard GNNs, edges in PID-Net v5 evolve:

```python
# Compute edge updates from node feature similarity + PID signals
node_sim = X_new @ X_new.T / sqrt(d)    # cosine similarity between nodes
edge_birth = sigmoid(W_birth @ concat(Xi_pair, Xd_pair))  # should new edge form?
edge_death = sigmoid(W_death @ concat(Xi_pair, Xd_pair))  # should edge dissolve?

# Update adjacency (differentiable)
A_new = A * (1 - edge_death) + (1 - A) * edge_birth

# Sparsify: only keep top-k edges per node (for efficiency)
A_new = top_k_sparse(A_new, k=k_edges)
```

**Key insight:** Edge birth/death is driven by the I-stream (historical patterns → persistent connections) and D-stream (novel relationships → new edges). The P-stream provides current context but doesn't directly control topology — you need HISTORY and CHANGE to know which connections matter.

---

## 4. Mathematical Formulation

### 4.1 State Space

Let the cognitive state at time t be:

```
Sₜ = (Xₜ, Aₜ, Wₜ, X̂ₜ)

where:
  Xₜ ∈ ℝ^{N×d}     — node features
  Aₜ ∈ [0,1]^{N×N}  — soft adjacency matrix
  Wₜ ∈ ℝ^{d×d}      — fast weight memory
  X̂ₜ ∈ ℝ^{N×d}     — predicted next state
```

### 4.2 PID Rewriting Rule (One Step)

```
P-stream:   Pₜ = σ(Aₜ · Xₜ · Wₘₛₘ + bₚ)                    # message passing
I-stream:   Iₜ = Wₜ · (Wq · Xₜ)                              # fast weight readout
D-stream:   Dₜ = Wₑ · (Xₜ - X̂ₜ)                             # prediction error
Gate:       gₜ = softmax(Wg · [mean(Pₜ); mean(Iₜ); mean(Dₜ)])# rule blend
Skip:       sₜ = σ(Ws · [mean(Pₜ); mean(Iₜ); mean(Dₜ)])     # skip probability

# Node update
Xₜ' = sₜ · Xₜ + (1-sₜ) · (gₜ[0]·Pₜ + gₜ[1]·Iₜ + gₜ[2]·Dₜ)

# Memory update (D→I coupling)
ηₜ = σ(Wη · Dₜ)                      # write strength from prediction error
kₜ = Wk · Xₜ,  vₜ = Wv · Xₜ'
Wₜ' = λ · Wₜ + ηₜ · (kₜ ⊗ vₜ)       # decay old + write new

# Prediction update
X̂ₜ₊₁ = Wp · Xₜ'                      # predict next state

# Edge update
Aₜ' = evolve_edges(Aₜ, Xₜ', Iₜ, Dₜ)  # topology evolution
```

### 4.3 Hamiltonian Energy Conservation

To prevent collapse, we define a Hamiltonian:

```
H(Xₜ, Wₜ) = ½||Xₜ||² + ½||Wₜ||²_F + V(Xₜ, Aₜ)

where V(Xₜ, Aₜ) = -Σᵢⱼ Aᵢⱼ · xᵢᵀxⱼ  (graph energy)
```

The PID update preserves total energy (within ε tolerance), enforced via:

```
# After PID update, project back to energy-conserving manifold
E_before = H(Xₜ, Wₜ)
E_after = H(Xₜ', Wₜ')
scale = sqrt(E_before / E_after)
Xₜ' *= scale  # renormalize to conserve energy
```

This mathematically prevents:
- **Collapse:** Energy can't go to zero (nodes can't all become identical)
- **Explosion:** Energy can't diverge (representations stay bounded)
- **Repetition:** Requires energy to sustain a fixed point, which gets redirected

### 4.4 Complexity Analysis

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| P-stream (message passing) | O(\|E\| · d) | Sparse, \|E\| = k·N with k edges/node |
| I-stream (fast weight read/write) | O(N · d²) | Matrix-vector multiply |
| D-stream (prediction error) | O(N · d) | Element-wise |
| Gate computation | O(d) | Global mean + linear |
| Edge evolution | O(N · k · d) | Only update k-nearest edges |
| **Total per rewriting step** | **O(N · d² + \|E\| · d)** | |
| **Per token (R steps)** | **O(R · (N · d² + k · N · d))** | |

For context window T with k edges per node and R rewriting steps:
- **PID-Net v5:** O(T · R · (d² + k·d)) per token → O(T²·R·(d² + k·d)) total
- **Transformer:** O(T² · d) total
- **Mamba/RWKV:** O(T · d²) total

With bounded graph size (only keep last C nodes via eviction):
- **PID-Net v5 (bounded):** O(C · R · (d² + k·d)) per token → **O(T · C · R)** total
- This is **O(T)** when C, R are constants — true linear scaling!

---

## 5. The PID Rewriting Rule — Detailed

### 5.1 Rule Variants (Mixture of Rewriting Rules)

Instead of one rewriting rule, maintain K candidate rules:

```python
class RewritingRule(nn.Module):
    def __init__(self, d_model, rule_id):
        self.W_msg = nn.Linear(d_model, d_model)   # message weights
        self.W_key = nn.Linear(d_model, d_model)    # memory key projection
        self.W_val = nn.Linear(d_model, d_model)    # memory value projection
        self.W_query = nn.Linear(d_model, d_model)  # memory query projection
        self.W_pred = nn.Linear(d_model, d_model)   # prediction weights
        self.W_err = nn.Linear(d_model, d_model)    # error projection
```

The gate selects a convex combination of rules:

```python
rule_logits = W_rule_select @ gate_input  # [K]
rule_weights = softmax(rule_logits)       # which rules to blend
X_new = sum(rule_weights[k] * rules[k](G) for k in range(K))
```

This is analogous to Wolfram's observation that different rewriting rules produce different physics. Our model learns WHICH rules produce useful cognition.

### 5.2 Graph Growth and Pruning

As the sequence grows, the graph grows. To maintain efficiency:

**Node eviction (bounded context):**
```python
# Score each node by "importance" (how much other nodes attend to it)
importance = A.sum(dim=0)  # [N] — total incoming edge weight
# Remove least important nodes when N > max_nodes
evict_idx = importance.argsort()[:N - max_nodes]
X = delete_rows(X, evict_idx)
A = delete_rows_cols(A, evict_idx)
```

**But store evicted nodes in fast weight memory:**
```python
# Before eviction, write to long-term memory
for idx in evict_idx:
    k = W_evict_key @ X[idx]
    v = W_evict_val @ X[idx]
    W_fast += outer(k, v)  # evicted node lives on in fast weights
```

This creates a natural **working memory** (active graph nodes) + **long-term memory** (fast weights) architecture.

### 5.3 Multi-Timescale Memory Decay

Following our power-law discovery, fast weight memory uses multi-timescale decay:

```python
# Instead of single λ, use multiple decay rates with power-law spacing
lambdas = [1 - 1/tau for tau in [2, 4, 8, 16, 32, 64, 128, 256]]  # power of 2 spacing

# Maintain separate fast weight banks per timescale
W_banks = [W_fast_1, W_fast_2, ..., W_fast_K]

# Each bank decays at its own rate
for k, (W, lam) in enumerate(zip(W_banks, lambdas)):
    W_banks[k] = lam * W + eta * outer(key, value)

# Read: query all banks, weighted sum
retrieved = sum(alpha[k] * W_banks[k] @ query for k in range(K))
# alpha learned — model decides which timescale matters
```

---

## 6. Multi-Scale Fractal Structure

### 6.1 The Sierpiński Principle

The same rewriting rule applied at every scale produces self-similar structure. For PID-Net v5:

```
Level 0 (Token):    Each character/token is a node
                    PID rewriting operates on individual tokens
                    Captures: phonemes, subwords, local syntax

Level 1 (Chunk):    Groups of ~16-64 tokens form super-nodes
                    PID rewriting operates on chunk representations
                    Captures: phrases, clauses, local semantics

Level 2 (Block):    Groups of ~4-16 chunks form block-nodes
                    PID rewriting operates on block representations
                    Captures: paragraphs, topic structure, discourse

Level 3 (Document): Groups of blocks form document-nodes
                    PID rewriting operates on document representations
                    Captures: themes, narrative arc, global coherence
```

### 6.2 Cross-Scale Communication

```
Token Graph ──(pool)──▶ Chunk Graph ──(pool)──▶ Block Graph
     ▲                      ▲                      │
     │                      │                      │
     └──(broadcast)─────────└──(broadcast)─────────┘
```

**Pool (bottom-up):** Aggregate node features within a chunk to create chunk-level node
**Broadcast (top-down):** Chunk-level decisions influence token-level processing

```python
# Bottom-up: token → chunk
chunk_features = pool(token_features, chunk_boundaries)  # learned pooling
chunk_graph = build_graph(chunk_features)
chunk_graph = pid_rewrite(chunk_graph, shared_rule)       # SAME rule!

# Top-down: chunk → token  
token_context = broadcast(chunk_features, chunk_boundaries)
token_features = token_features + W_topdown @ token_context
```

**The shared rule is the fractal property.** The SAME PID rewriting rule (same weights) operates at every scale. Different behavior emerges from different graph structures at each scale — not from different learned rules.

### 6.3 Adaptive Chunk Boundaries

Chunk boundaries aren't fixed — they're determined by the D-stream:

```python
# High prediction error = surprise = boundary
error_magnitude = ||X_actual - X_predicted||²  # per token
boundary_prob = sigmoid(W_boundary @ error_magnitude)  # probability of chunk break

# Use Gumbel-softmax for differentiable boundary selection
boundaries = gumbel_softmax(boundary_prob, hard=True)
```

This means the model automatically segments text into meaningful units based on where SURPRISE occurs — exactly how humans parse language (we chunk at phrase boundaries, topic shifts, etc.).

---

## 7. Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2) — "Prove the Graph"

**Goal:** Validate that PID rewriting on a graph produces stable, non-collapsing training.

**Build:**
- `HypergraphState`: Data structure for (nodes, edges, fast_weights, prediction)
- `PIDRewriteStep`: Single rewriting step with P/I/D streams
- `PIDGraphNet`: Stack of R rewriting steps with input/output projections
- Fixed graph topology (sequential chain: each token connects to previous k tokens)
- No edge evolution yet, no multi-scale, no adaptive compute

**Test on:**
- Shakespeare character-level (same as v3, for comparison)
- Synthetic associative recall (can the fast weight I-stream retrieve past patterns?)

**Success criteria:**
- Training loss decreases monotonically
- No repetition collapse (test generation every epoch)
- P, I, D gates all have non-trivial values (none dies below 0.05)
- Fast weight I-stream shows non-zero retrieval (compare with vs. without)

**Params:** ~5M (comparable to v3 for fair comparison)

### Phase 2: Dynamic Topology (Weeks 3-4) — "Let It Breathe"

**Goal:** Enable the graph to evolve — edges born, strengthened, weakened, removed.

**Add:**
- `EdgeEvolver`: Differentiable edge birth/death from I/D signals
- Top-k sparsification (keep only k most important edges per node)
- Node eviction with memory consolidation (evicted nodes → fast weights)
- Bounded context window (max N nodes in active graph)

**Test on:**
- Shakespeare (does dynamic topology improve over fixed?)
- Copy task (can the model learn to create direct edges between copied tokens?)
- Selective copy (does edge evolution create meaningful connections?)

**Success criteria:**
- Learned graph topology differs from sequential chain
- Edge patterns correlate with linguistic structure (e.g., edges between subject-verb)
- Performance ≥ Phase 1 on language modeling
- Memory consolidation preserves information (test: query evicted nodes via fast weights)

### Phase 3: Predictive Coding D-Stream (Weeks 5-6) — "Surprise-Driven"

**Goal:** Replace finite-difference D with true prediction error. Add D→I coupling.

**Add:**
- `PredictiveCoder`: Predicts next graph state, computes error
- D→I coupling: prediction error magnitude → fast weight write strength
- Adaptive compute: high error → more rewriting steps, low error → fewer

**Test on:**
- Language modeling (Shakespeare + TinyStories)
- Anomaly detection: insert random tokens, verify D-stream spikes
- Measure: does the predictor improve over training? (error should decrease)

**Success criteria:**
- D-stream prediction error decreases over training (model learns to predict)
- D-stream spikes on genuinely surprising tokens (proper nouns, rare words)
- D→I coupling: surprising events are recalled better than routine ones
- Adaptive compute: boring passages use fewer steps than complex ones

### Phase 4: Multi-Scale Fractal (Weeks 7-10) — "The Sierpiński"

**Goal:** Same PID rule operating at token, chunk, and block levels.

**Add:**
- `FractalPIDNet`: Multi-scale with shared rewriting rule
- Adaptive chunk boundaries from D-stream
- Cross-scale communication (pool up, broadcast down)
- Power-law multi-timescale memory banks

**Test on:**
- TinyStories (needs multi-sentence coherence)
- Long-range dependency tasks (do block-level representations help?)
- Chunk boundary analysis (do learned boundaries match linguistic units?)

**Success criteria:**
- Multi-scale model outperforms single-scale on coherence metrics
- Chunk boundaries correlate with syntactic/semantic boundaries
- Power-law decay outperforms exponential decay on long sequences
- Shared rule produces different gate patterns at different scales

### Phase 5: Scale & Compete (Weeks 11-16) — "Prove It"

**Goal:** Scale to meaningful size and benchmark against established architectures.

**Build:**
- Scale to 125M → 350M parameters
- Optimize: kernel fusion, sparse ops, memory-efficient backprop
- Train on standard benchmarks (OpenWebText, The Pile subset)

**Test against:**
- Transformer (same param count)
- Mamba (same param count)
- RWKV (same param count)

**Success criteria:**
- See [Section 8: Expectations](#8-expectations--milestones)

---

## 8. Expectations & Milestones

### 8.1 Easy (High Confidence — 80%+ probability)

These should work based on established theory:

| Expectation | Basis | Milestone |
|-------------|-------|-----------|
| PID rewriting doesn't collapse | Hamiltonian conservation prevents it | Phase 1 |
| Fast weight I-stream outperforms EMA | EMA is provably lossy; FW is retrievable | Phase 1 |
| Gate values differentiate (P≠I≠D) | v3 showed this even with worse architecture | Phase 1 |
| Dynamic edges outperform fixed topology | Information theory: learned structure > imposed | Phase 2 |
| D-stream prediction error decreases over training | Gradient descent on squared error | Phase 3 |
| Multi-scale outperforms single-scale on long sequences | Hierarchical processing is provably more efficient | Phase 4 |

### 8.2 Normal (Medium Confidence — 50-70%)

Likely to work but may need iteration:

| Expectation | Challenge | Milestone |
|-------------|-----------|-----------|
| Match Transformer perplexity at same param count | Transformers are heavily optimized; graph ops are new | Phase 5 |
| Learned topology shows linguistic structure | Discovering structure from data is hard | Phase 2-3 |
| Adaptive chunk boundaries match syntactic units | Unsupervised parsing is an open problem | Phase 4 |
| D→I coupling improves recall of surprising events | Coupling strength needs careful tuning | Phase 3 |
| Adaptive compute provides >2x speedup on easy text | Skip gate needs to learn "what's easy" | Phase 3-5 |
| Model develops distinct "cognitive modes" via gate patterns | Requires sufficient capacity and diverse training | Phase 5 |

### 8.3 Exceptional (Low Confidence — 10-30%, but would be groundbreaking)

The moonshots:

| Expectation | Why it's hard | Why it might work | Milestone |
|-------------|---------------|-------------------|-----------|
| Beat Transformer at same FLOPS (not just params) | Graph ops have overhead | Sparsity + adaptive compute could compensate | Phase 5+ |
| Emergent multi-step reasoning without CoT | No architecture has achieved this | Rewriting steps = implicit reasoning chain | Phase 5+ |
| Transfer learning from rewriting rules | Rules are abstract; might not transfer | Rules are domain-agnostic by design | Phase 5+ |
| Fractal rule sharing reduces params 4x with same performance | Sharing weights is aggressive | Same rule at different scales IS the hypothesis | Phase 4-5 |
| Self-organizing memory hierarchy (no designed levels) | Hierarchy usually needs design | Wolfram's physics suggests it emerges | Phase 5+ |
| Graph topology reveals interpretable reasoning chains | Neural nets are famously opaque | Graph edges are explicit, inspectable | Phase 2+ |

---

## 9. Testing & Debugging Framework

### 9.1 Unit Tests (run on every commit)

```python
class TestPIDRewrite:
    def test_energy_conservation():
        """Hamiltonian energy within ε after rewrite step"""
        
    def test_no_nan_gradients():
        """All parameters receive finite gradients"""
        
    def test_gate_values_valid():
        """Gates sum to 1, all >= 0"""
        
    def test_fast_weight_write_read():
        """Store a pattern, retrieve it, cosine_sim > 0.9"""
        
    def test_prediction_error_decreases():
        """After 100 steps of same input, prediction error → 0"""
        
    def test_edge_evolution_differentiable():
        """Gradients flow through edge birth/death"""
        
    def test_node_eviction_preserves_memory():
        """Evicted node retrievable from fast weights"""
        
    def test_skip_gate_identity():
        """When skip=1, output equals input exactly"""
```

### 9.2 Integration Tests (run before each phase milestone)

```python
class TestArchitecture:
    def test_no_repetition_collapse():
        """Generate 500 tokens from 5 prompts, no repeated n-grams > 5"""
        
    def test_all_streams_active():
        """P, I, D gates all > 0.05 averaged over validation set"""
        
    def test_loss_decreases():
        """Training loss decreases monotonically over 1000 steps"""
        
    def test_generation_diversity():
        """10 generations from same prompt have < 0.5 BLEU with each other"""
        
    def test_graph_not_degenerate():
        """Adjacency matrix is not all-ones or all-zeros"""
        
    def test_memory_utilization():
        """Fast weight matrix has rank > d/4 after training"""
```

### 9.3 Diagnostic Dashboard (real-time during training)

Track and log every epoch:

```
═══════════════════ PID-Net v5 Diagnostics ═══════════════════
Epoch: 12/100    Loss: 2.341    BPB: 0.891

Gate Values (mean ± std):
  P-gate:  0.35 ± 0.08    [████████░░░░░░░░]
  I-gate:  0.27 ± 0.12    [██████░░░░░░░░░░]
  D-gate:  0.38 ± 0.09    [█████████░░░░░░░]
  Skip:    0.15 ± 0.20    [███░░░░░░░░░░░░░]

Memory:
  Fast weight rank:     48/64 (75%)
  Write strength (η):   0.23 ± 0.15
  Retrieval cos_sim:    0.67

Prediction:
  D-stream error:       0.45 (↓ from 1.2 at epoch 1)
  Predictor accuracy:   62%

Graph:
  Active nodes:         128/128
  Avg edges/node:       8.3
  Edge birth rate:      0.12/step
  Edge death rate:      0.09/step
  Graph diameter:       4.2

Generation Sample:
  Prompt: "To be or not"
  Output: "to be, that is the question whether..."   ✅

Collapse Check: 🟢 OK (0/5 prompts show repetition)
═══════════════════════════════════════════════════════════════
```

### 9.4 Debugging Playbook

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| I-gate dies (< 0.05) | Fast weights not storing useful info | Check write strength η, ensure D→I coupling works |
| D-gate dominates (> 0.7) | Prediction is terrible, all error | Warm up predictor separately before full training |
| Repetition collapse | Energy not conserved, PID fixed point | Check Hamiltonian constraint, increase kickout strength |
| Graph becomes fully connected | Edge birth >> death | Increase death rate, lower birth threshold |
| Graph becomes disconnected | Edge death >> birth | Lower death threshold, initialize with chain |
| Loss plateaus early | Not enough rewriting steps | Increase R, or allow adaptive R |
| Training loss NaN | Gradient explosion in fast weights | Clip fast weight norm, reduce write strength |
| All nodes converge to same vector | Message passing over-smoothing | Add residual connections, reduce message passing steps |
| Skip gate always 1 (doing nothing) | Network found lazy solution | Initialize skip bias negative (-2), add skip penalty to loss |

---

## 10. Future Expansion

### 10.1 Near-Term (v5.1 — v5.3)

**v5.1: Hyperedge Support**
- Add true hyperedges (connecting 3+ nodes)
- Captures higher-order relationships (e.g., subject-verb-object triples)
- Implementation: Tensor decomposition for hyperedge computation

**v5.2: Mixture of Rewriting Rules (MoR)**
- K different rewriting rules, gate selects per-step
- Analogous to Mixture of Experts, but for graph operations
- Different rules for different "cognitive tasks" (pattern matching vs. memory recall vs. novelty detection)

**v5.3: Continuous-Time PID**
- Replace discrete rewriting steps with Neural ODE
- PID dynamics as continuous differential equations
- Adaptive step size = adaptive compute naturally
- Connects to Hamiltonian Neural Networks literature

### 10.2 Medium-Term (v6)

**v6: Self-Modifying Rules**
- The rewriting rule itself evolves during training/inference
- Meta-learning: learn to learn new rules
- Rules stored in a "rule memory" — fast weight programmer for rules
- The model can discover new computational primitives

### 10.3 Long-Term (v7+)

**v7: True Ruliad Search**
- Explore the space of ALL possible rewriting rules
- The model navigates the "ruliad" (Wolfram's term for the entangled limit of all computation)
- Intelligence = finding the right path through the ruliad
- This is where AGI might live

**v8: Multi-Agent Rewriting**
- Multiple PID-Net agents, each maintaining their own hypergraph
- Agents communicate by sharing graph fragments
- Collective intelligence from interacting rewriting systems
- Social cognition, theory of mind

---

## 11. Risk Analysis

### 11.1 Technical Risks

| Risk | Severity | Probability | Mitigation |
|------|----------|-------------|------------|
| Graph ops too slow | Medium | 25% | MLX unified memory eliminates transfer overhead; custom Metal kernels if needed |
| Backprop through topology changes unstable | High | 30% | Soft/differentiable edges, gradient clipping |
| Over-smoothing (all nodes converge) | Medium | 50% | Residual connections, normalization, limited message passing |
| Fast weight memory explodes | Medium | 30% | Norm clipping, decay, capacity limiting |
| Architecture too complex to debug | Medium | 60% | Phased rollout, extensive diagnostics |
| Doesn't scale beyond toy tasks | High | 35% | Phase 5 designed specifically to test this |

### 11.2 Research Risks

| Risk | Severity | Probability | Mitigation |
|------|----------|-------------|------------|
| Graph rewriting adds overhead without benefit | High | 25% | Compare against fixed-topology ablation at each phase |
| Fractal sharing hurts (scales need different rules) | Medium | 40% | Ablate: shared vs. independent rules per scale |
| Prediction error is too noisy for useful D signal | Medium | 30% | Try EMA of error, multiple prediction horizons |
| Adaptive compute too hard to learn | Low | 20% | Pre-train with fixed compute, then unlock skip gate |

### 11.3 Go/No-Go Decision Points

After each phase, evaluate:

1. **Phase 1 → Phase 2:** Does PID rewriting match or beat v3? If no, debug before proceeding.
2. **Phase 2 → Phase 3:** Does dynamic topology help? If no, consider fixing topology and skipping to Phase 3.
3. **Phase 3 → Phase 4:** Does predictive coding D-stream work? If no, fall back to finite difference.
4. **Phase 4 → Phase 5:** Does fractal structure help? If no, use single-scale for Phase 5.
5. **Phase 5 → Scale:** Are we within 20% of Transformer baseline? If no, fundamental rethink needed.

---

## 12. Technical Challenges & Solutions

### 12.1 Challenge: Differentiable Graph Topology

**Problem:** Adding/removing edges is a discrete operation. Discrete = not differentiable = no gradients.

**Solution:** Soft edges. Every possible edge exists with a weight ∈ [0, 1]. "Removing" an edge = weight → 0. "Adding" = weight → 1. Top-k sparsification uses straight-through estimator for gradients.

```python
# Differentiable top-k via Gumbel-softmax
edge_logits = compute_edge_scores(X)           # [N, N]
edge_weights = gumbel_softmax(edge_logits, tau=0.5, hard=True)  # differentiable
A = edge_weights * sigmoid(edge_logits)        # combine selection with magnitude
```

### 12.2 Challenge: Sequence → Graph Mapping

**Problem:** Language is sequential. How does a graph representation help?

**Solution:** The graph captures relationships that AREN'T sequential. Natural language has long-range dependencies (coreference, topic coherence) that a sequential model must propagate step-by-step, but a graph can capture directly via edges.

```
Sequential:  "The cat that sat on the mat was happy"
             The → cat → that → sat → on → the → mat → was → happy
             (9 hops from "cat" to "happy")

Graph:       "cat" ──edge── "happy" (direct connection learned via PID)
             (1 hop)
```

### 12.3 Challenge: Compute Efficiency

**Problem:** Graph operations are irregular — hard to parallelize on traditional CUDA GPUs.

**Solution:** MLX on Apple Silicon — architectural alignment.

Why this works better than CUDA for our specific architecture:
1. **Unified memory** — No CPU↔GPU transfers for irregular graph access patterns. Sparse gather/scatter operations that kill CUDA performance are near-free on Apple Silicon.
2. **Dynamic shapes** — MLX handles changing graph topology natively. No recompilation, no shape guards, no padding waste.
3. **Lazy evaluation** — Automatic fusion of many small per-node/per-edge operations into efficient Metal kernels.
4. **Memory bandwidth** — Apple Silicon's bandwidth per dollar exceeds NVIDIA for memory-bound workloads (which sparse graph ops ARE).

Additional strategies:
1. **Fixed maximum graph size** with masking (for batch efficiency)
2. **Block-sparse adjacency** (dense within local neighborhoods, sparse globally)
3. **Custom Metal kernels** for critical path operations (if needed at scale)

**Scaling Path:**
- **Phases 1-4:** MacBook (M-series) — single device, ≤5M params
- **Phase 5:** Mac Studio (M Ultra) — 192GB+ unified memory, 125-350M params
- **Phase 5+:** Mac Studio cluster — MLX distributed training, 1B+ params
- **Phase 6+:** Multi-node Apple Silicon — custom cluster, 10B+ params
- **Advantage:** PID-Net v5's sparse graph architecture is fundamentally better suited to unified memory than to CUDA's separated memory model. We're not competing with NVIDIA on dense matmuls — we're playing a different game on hardware optimized for that game.

### 12.4 Challenge: Multi-Scale Synchronization

**Problem:** Token-level updates happen every step, but chunk-level updates happen every 16-64 steps. How to synchronize?

**Solution:** Asynchronous updates with a sync buffer:
```python
# Token level: runs every step
token_graph = pid_rewrite(token_graph)

# Every chunk_size steps: update chunk level
if step % chunk_size == 0:
    chunk_graph = pool(token_graph)
    chunk_graph = pid_rewrite(chunk_graph)  # same rule!
    token_context = broadcast(chunk_graph)
    token_graph.nodes += token_context      # top-down influence
```

---

## 13. Appendix: Theoretical Foundations

### 13.1 Wolfram Physics → PID-Net Mapping

| Wolfram Physics | PID-Net v5 |
|-----------------|------------|
| Hypergraph | Cognitive graph (nodes = features, edges = relationships) |
| Rewriting rule | Learned PID update function |
| Rule application | One PID rewriting step |
| Spatial hypergraph | P-stream (current state) |
| Causal graph | D-stream (what caused what to change) |
| Multiway system | Gate (exploring multiple rewriting paths) |
| Branchial space | I-stream (superposition of past states in memory) |
| Ruliad | Space of all possible PID rules (future v7) |

### 13.2 Neuroscience Correspondence

| Brain | PID-Net v5 |
|-------|------------|
| Cortical columns | Graph nodes |
| Synaptic connections | Graph edges |
| Synaptic plasticity (LTP/LTD) | Edge birth/death |
| Working memory (~7 items) | Active graph nodes (bounded) |
| Long-term memory | Fast weight matrix |
| Dopamine (learning rate modulation) | D→I coupling (error → write strength) |
| Predictive coding (top-down predictions) | D-stream prediction error |
| Attention (selective enhancement) | Gate + edge strengthening |
| Sleep consolidation | Memory compression during eviction |

### 13.3 Control Theory Correspondence

| PID Control | PID-Net v5 |
|-------------|------------|
| Plant (system being controlled) | The hypergraph itself |
| Setpoint | Predicted next state (X̂) |
| Error signal | Prediction error (X - X̂) |
| Proportional term | P-stream (react to current state) |
| Integral term | I-stream (accumulated memory) |
| Derivative term | D-stream (rate of change / prediction error) |
| Controller output | Gate-weighted blend of P, I, D |
| Stability (no oscillation) | Hamiltonian energy conservation |
| Adaptive control | Learned rewriting rules that update |

### 13.4 Key References

1. Wolfram, S. (2020). "A Project to Find the Fundamental Theory of Physics"
2. Schmidhuber, J. (1992). "Learning to Control Fast-Weight Memories"
3. Schlag, I. et al. (2021). "Linear Transformers Are Secretly Fast Weight Programmers"
4. Ramsauer, H. et al. (2021). "Hopfield Networks is All You Need"
5. Sun, Y. et al. (2024). "Learning to (Learned) Hash: TTT Layers"
6. Rao, R. & Ballard, D. (1999). "Predictive Coding in the Visual Cortex"
7. Bengio, Y. (2017). "The Consciousness Prior"
8. Gregor, K. & Biny, F. (2019). "Temporal Difference Variational Auto-Encoder"
9. Anderson, J.R. (2007). "How Can the Human Mind Occur in the Physical Universe?" (ACT-R, power-law decay)

---

## Change Log

| Date | Version | Changes |
|------|---------|---------|
| 2026-03-14 | 0.1 | Initial specification |

---

*"The universe is not a collection of objects, but a network of relationships. Intelligence is not a function, but an evolving pattern."*
