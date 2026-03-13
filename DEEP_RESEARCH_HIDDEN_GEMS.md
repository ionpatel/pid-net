# Deep Research II: The Hidden Gems — Overlooked Ideas That Could Change Everything
### Forgotten Papers, Abandoned Architectures, and Underexplored Concepts
### Ion × Harshil | March 2026

---

## Why This Matters

The AI field suffers from **survivorship bias**. Transformers won, so everything else got abandoned. But "won" doesn't mean "optimal" — it means "scaled first with available hardware." Many abandoned ideas were architecturally superior but couldn't run on GPUs efficiently in 2017.

The hardware has changed. The math hasn't. Time to revisit the dead.

---

## 1. Test-Time Training (TTT) — The Model That Learns During Inference

### Paper: Sun et al., "Learning to (Learn at Test Time)" — Stanford/Meta, 2024

### What It Is
TTT replaces the hidden state of an RNN with the **weights of a small neural network**. During inference, each "layer" performs a mini training step — the model literally learns from the sequence as it processes it.

```
Standard RNN:  h_t = f(h_{t-1}, x_t)           # update hidden state
TTT Layer:     W_t = W_{t-1} - η∇L(W_{t-1}, x_t)  # update WEIGHTS via gradient descent
Output:        y_t = g(W_t, x_t)                # use updated weights for prediction
```

### Why It's Revolutionary
- **Infinite effective context:** The model doesn't just "remember" — it LEARNS from the context. Every new token updates the model's weights.
- **Self-improvement at inference:** The model gets better at the specific task/text it's processing, in real-time.
- **Matches Mamba at 1.3B params** with unbounded context.
- **Fundamentally different from attention:** Attention RETRIEVES from context. TTT LEARNS from context.

### Why It Was Overlooked
- Training is expensive (gradient computation inside the forward pass)
- Requires careful learning rate scheduling per-layer
- Published in 2024, still early

### **PID-Net Connection: ⭐⭐⭐⭐⭐ GAME-CHANGING**

This is what our I-stream SHOULD be. Not a static EMA. Not a fixed matrix memory. A **learning system** that updates its own weights based on what it sees.

```
I-stream as TTT:
  W_I(t) = W_I(t-1) - η_I · D(t) · ∇L(W_I(t-1), x_t)
  
  Where:
  - W_I are the I-stream's weights (they CHANGE during inference)
  - η_I is modulated by D-gate (surprise → learn faster)
  - L is a self-supervised loss (predict next input)
```

The D-gate controls the learning rate: high surprise → learn aggressively (this is new/important). Low surprise → barely update (already know this). 

**This is EXACTLY what dopamine does in the brain** — modulates learning rate based on prediction error.

---

## 2. Modern Hopfield Networks — Attention IS Associative Memory

### Paper: Ramsauer et al., "Hopfield Networks is All You Need" — 2020

### What It Is
Classical Hopfield networks (1982) store binary patterns and retrieve the nearest stored pattern. Modern Hopfield Networks extend this to continuous patterns with **exponential storage capacity**.

The key theorem: **Transformer attention is equivalent to one step of a Modern Hopfield Network update.**

```
Attention:     softmax(Q·K^T/√d) · V
Hopfield:      softmax(β · x · Ξ^T) · Ξ      # Ξ = stored patterns

These are the SAME operation.
```

### Why It's Revolutionary
- **Transformers are memory retrieval systems.** Every attention layer retrieves stored patterns. This isn't a metaphor — it's mathematical equivalence.
- **Exponential storage:** Modern Hopfield networks can store exponentially many patterns (2^(d/2) patterns in d dimensions).
- **Energy-based:** Hopfield networks have a well-defined energy function that decreases with each update. Convergence is guaranteed.
- **Associative memory:** Content-addressable — retrieve by similarity, not by index.

### Why It Was Overlooked
- Published 2020 but overshadowed by scaling papers
- Viewed as "theoretical" rather than practical
- Hopfield networks carry baggage from the 1980s (old = dismissed)

### **PID-Net Connection: ⭐⭐⭐⭐⭐ CRITICAL**

If attention = Hopfield memory retrieval, then the I-stream should be a **Hopfield network**:

```
I-stream as Modern Hopfield:
  Store:    Ξ_t = [Ξ_{t-1}; new_pattern]  # add new patterns (controlled by I-gate)
  Retrieve: y = softmax(β · query · Ξ^T) · Ξ  # retrieve by similarity
  Forget:   Ξ_t = decay(Ξ_{t-1})          # forget old patterns (controlled by D-gate)
```

This gives us:
- **Exact retrieval** (not blurry EMA)
- **Energy-based dynamics** (guaranteed convergence, no collapse)
- **Content-addressable memory** (query what you need)
- **Mathematically principled** forgetting (energy landscape reshaping)

**The I-stream wouldn't die because it would be USEFUL — it provides exact retrieval, which the model actually needs.**

---

## 3. Differential Transformer — The D-Stream Applied to Attention

### Paper: Ye et al., "Differential Transformer" — Microsoft Research, 2024

### What It Is
Standard attention computes: `softmax(QK^T)V`

Differential attention computes:
```
Attn = softmax(Q₁K₁^T) - λ · softmax(Q₂K₂^T)
Output = Attn · V
```

It takes the **DIFFERENCE** between two attention patterns. This cancels out noise and attends only to truly relevant tokens.

### Why It's Revolutionary
- **Reduces hallucination** by 30-50% (noise cancellation)
- **Better in-context learning** (cleaner signal)
- **Matches or beats standard attention** at same param count
- **Mathematically principled:** Like differential amplifiers in electronics — subtract common-mode noise

### Why It Matters for Us
**This IS a D-stream applied to attention.** They're computing the derivative/difference of attention patterns. They proved it works. We should generalize this:

```
PID-Attention:
  Attn_P = softmax(Q_P · K^T)                    # standard attention (current)
  Attn_I = softmax(Q_I · K_accumulated^T)         # attention over accumulated context
  Attn_D = Attn_P(t) - Attn_P(t-1)               # differential attention (what changed)
  
  Output = g_P · Attn_P · V + g_I · Attn_I · V + g_D · Attn_D · V
```

---

## 4. Sparse Distributed Memory (SDM) — Kanerva's Brain Memory Model

### Paper: Kanerva, "Sparse Distributed Memory" — 1988 (yes, 1988)

### What It Is
SDM models how the brain stores and retrieves memories in a high-dimensional binary space. Key idea: memory is stored not in specific neurons but **distributed across many neurons** in overlapping patterns.

```
Address space: {0,1}^n  (n ≈ 1000 dimensions)
Hard locations: M random points in address space (M << 2^n)
Write: activate all hard locations within Hamming distance r of address
Read: sum contributions from all activated hard locations
```

### Why It's Revolutionary (and Forgotten)
- **Robust to noise:** Corrupted addresses still retrieve the right memory
- **Graceful degradation:** Performance degrades smoothly as memory fills
- **Biological plausibility:** Maps to how cerebellar/cortical circuits work
- **Interference patterns:** Similar to how real memories interfere/blend
- **Natural generalization:** Similar inputs retrieve similar memories (no explicit similarity search)

### Why It Was Forgotten
- 1988 — before modern deep learning
- Binary representations seemed limiting
- GPU-unfriendly in its original form

### **PID-Net Connection: ⭐⭐⭐⭐ HIGH**

SDM is the I-stream memory model we need. Not exact key-value pairs (too rigid), not EMA (too lossy), but **distributed associative storage** with natural generalization.

Modern SDM implementation using continuous representations:
```
I-stream as SDM:
  Address = P_t (current perception → used as memory address)
  Write:  M += outer(address, value) * I_gate  # distributed write
  Read:   y = M^T · softmax(M · address)       # attention-like retrieval
  Forget: M *= (1 - D_gate * forget_rate)       # D-modulated forgetting
```

---

## 5. Neural Turing Machines & Differentiable Neural Computers

### Papers: 
- Graves et al., "Neural Turing Machines" — DeepMind, 2014
- Graves et al., "Hybrid Computing Using a Neural Network with Dynamic External Memory" — DeepMind, 2016

### What They Are
NTMs augment neural networks with an external memory matrix and differentiable read/write heads. The network learns to:
1. **Address** memory (by content similarity or location)
2. **Read** from specific locations
3. **Write** to specific locations
4. **Erase** specific locations

DNCs extend NTMs with:
- **Temporal linking:** Track order of writes for sequential recall
- **Usage tracking:** Know which locations are free/used
- **Multiple read/write heads**

### Why They Were Revolutionary
- **Turing-complete** with sufficient memory
- **Learned algorithms:** NTMs/DNCs learned to sort, copy, and solve graph problems from examples
- **Explicit memory management** — unlike RNN hidden states, memory is transparent and inspectable
- **Variable-length memory** — can allocate more memory for harder problems

### Why They Were Abandoned
- **Training instability:** Discrete memory addressing is hard to differentiate through
- **Slow convergence:** 100x more training steps than LSTMs for same tasks
- **Didn't scale:** Work great on algorithmic tasks, failed on natural language at scale
- **Transformers arrived:** Attention provided implicit memory without explicit read/write

### Why We Should Revisit
- Training tricks have improved dramatically since 2016
- The memory management ideas (usage tracking, temporal linking) are exactly what transformers LACK
- **DNC's temporal link matrix** tracks the ORDER in which memories were written — this is a form of I-stream with sequence awareness

### **PID-Net Connection: ⭐⭐⭐⭐ HIGH**

DNC read/write heads map to PID gates:
```
Write gate → I-gate (what to store)
Erase gate → D-gate modulated forget (surprise-based erasure)
Read gate → P-gate (what to retrieve for current processing)
Temporal links → I-stream ordering (sequence of memories)
Usage vector → adaptive memory allocation
```

---

## 6. The Consciousness Prior — Bengio's Bottleneck Theory

### Paper: Bengio, "The Consciousness Prior" — 2017

### What It Is
Yoshua Bengio (Turing Award winner, "godfather of deep learning") proposed that consciousness is a **low-dimensional bottleneck** that forces the brain to form compact, communicable representations.

Key claims:
1. Only a few concepts are **consciously active** at any time (~7±2, Miller's law)
2. The "consciousness bottleneck" forces **disentangled representations** — each slot represents one concept
3. Unconscious processing is high-dimensional; consciousness compresses to low-dimensional
4. This bottleneck is what enables **language** (we can describe conscious states in words)
5. Attention IS the selection mechanism for what enters consciousness

### Why It's Profound
- **Explains why transformers work:** Attention = consciousness selection
- **Explains why MoE works:** Expert routing = unconscious specialization, router = conscious selection
- **Predicts:** Models with bottlenecks should generalize better (and they do — see information bottleneck theory)

### **PID-Net Connection: ⭐⭐⭐⭐⭐ CRITICAL**

**The gate IS the consciousness bottleneck.**

The 3-way PID gate forces the model to choose: "Am I reacting to NOW (P), drawing on MEMORY (I), or detecting CHANGE (D)?" This is a low-dimensional decision (3 values summing to 1) that compresses the model's cognitive mode.

Extension — **Consciousness slots:**
```
Instead of one gate per layer, have K consciousness slots:
  Slot 1: (P=0.8, I=0.1, D=0.1) — "I'm currently focused on pattern matching"
  Slot 2: (P=0.1, I=0.7, D=0.2) — "I'm recalling context"
  Slot 3: (P=0.2, I=0.1, D=0.7) — "I'm detecting something novel"

Output = Σ attention(query, slot_k) * PID_output(slot_k)
```

This gives the model **multiple simultaneous cognitive modes** — just like the brain can simultaneously perceive (P), remember (I), and detect novelty (D) for different aspects of the input.

---

## 7. Predictive Coding Networks — The Brain's Algorithm, Implemented

### Papers:
- Rao & Ballard, "Predictive Coding in Visual Cortex" — 1999
- Millidge et al., "Predictive Coding Approximates Backprop" — 2022
- Salvatori et al., "Associative Memories and Predictive Coding" — 2024

### What It Is
Predictive coding says each layer of the brain:
1. Predicts what the layer below will send (top-down prediction)
2. Receives actual input from below
3. Computes **prediction error** = actual - predicted
4. Sends only the error upward (not the full signal)
5. Updates its model to reduce future errors

```
ε_i = x_i - f_i(μ_{i+1})           # error = input - prediction
dμ_i/dt = ε_i - ∂ε_{i+1}/∂μ_i     # update beliefs
```

### Why It's Revolutionary
- **10-100x more efficient** than processing full signals (only errors propagate)
- **Can approximate backpropagation** using only local learning rules (Millidge 2022)
- **No separate training/inference phases** — the network learns continuously
- **Hierarchical:** Naturally creates a hierarchy of increasingly abstract representations
- **The brain actually does this** — neuroscience evidence is strong

### Why It Never Scaled
- Hard to implement efficiently on GPUs (sequential layer-wise updates)
- Convergence requires multiple iterations per input (settle to equilibrium)
- Transformer scaling was easier/faster with existing infrastructure

### **PID-Net Connection: ⭐⭐⭐⭐⭐ THIS IS PID-NET'S NATIVE ALGORITHM**

Predictive coding IS PID control:
```
P = current input (bottom-up signal)
I = accumulated beliefs about the world (μ, the internal model)
D = prediction error (ε = actual - predicted)
Gate = precision weighting (how much to trust each signal)
```

**The prediction error IS the D-stream.** We've been computing D as a finite difference of hidden states, but the RIGHT D is the prediction error — how wrong the model is about what comes next.

**Implementation:**
```python
class PredictivePIDLayer:
    def forward(self, x, prediction_from_above):
        P = self.encode(x)                           # bottom-up perception
        I = self.memory.read(P)                       # retrieve relevant memories
        prediction = self.predict(P, I)               # generate prediction
        D = x - prediction                            # prediction ERROR
        
        # Update memory based on surprise
        self.memory.write(P, strength=||D||)          # high error → strong write
        
        # Only propagate the error upward (not full signal)
        output = self.gate(P, I, D) * D               # gated error
        return output, prediction
```

This is 10-100x more efficient than standard processing because only errors (surprising/novel information) propagate through the network. Predictable input is handled locally.

---

## 8. Echo State Networks & Reservoir Computing — The Untrained Powerhouse

### Papers:
- Jaeger, "The Echo State Approach" — 2001
- Maass et al., "Liquid State Machines" — 2002

### What It Is
**The radical idea:** Don't train the recurrent part at all. Use a large random recurrent network (the "reservoir") as a fixed nonlinear dynamical system. Only train a linear readout layer.

```
h_t = tanh(W_res · h_{t-1} + W_in · x_t)    # fixed random weights (NOT trained)
y_t = W_out · h_t                              # only this is trained (linear regression!)
```

### Why It's Revolutionary
- **Training is trivially easy** — linear regression on the readout
- **Real-time learning** — can update readout incrementally
- **Computationally cheap** — no backprop through time
- **Surprisingly good** — matches trained RNNs on many time-series tasks
- **Theoretical depth:** The reservoir acts as a kernel — it projects inputs into a high-dimensional space where linear separation becomes possible

### Why It Was Overlooked
- **Can't compete at scale** — fixed random weights limit capacity
- **No representation learning** — reservoir can't learn task-specific features
- **GPU unfriendly** — sparse random recurrence

### **PID-Net Connection: ⭐⭐⭐ MODERATE but INSIGHTFUL**

What if PID-Net's P/I/D weight matrices are **fixed** (like a reservoir) and only the gates are trained?

```
W_p, W_i, W_d = fixed random (reservoir)
gate = trained (the only learnable part)
```

This would mean:
- P/I/D streams provide a rich, fixed nonlinear basis
- The gate learns WHICH basis elements to use for each input
- Training is dramatically simpler (only gate parameters)
- The "intelligence" is entirely in the gating — in the SELECTION, not the representation

This maps to the consciousness prior: the representations are unconscious (fixed, automatic), the selection mechanism is conscious (learned, deliberate).

---

## 9. Infini-Attention — Compressive Memory Within Attention

### Paper: Munkhdalai et al., "Leave No Context Behind" — Google, 2024

### What It Is
Infini-Attention adds a **compressive memory** to standard attention. Old key-value pairs don't get discarded when they leave the context window — they get compressed into a fixed-size memory matrix.

```
Standard attention: process only tokens within window
Infini-attention:   
  1. Standard attention within current segment
  2. Also query a compressive memory M from all previous segments
  3. M is updated incrementally: M += (keys^T · values) / sum(keys)
  4. Combine: output = gate * local_attention + (1-gate) * memory_retrieval
```

### Why It's Revolutionary
- **Infinite context with bounded memory** — truly unlimited sequence length
- **Drop-in replacement** for standard attention — minimal architectural change
- **Works at scale** — demonstrated with Gemini-scale models
- **The compression is learned** — not lossy averaging, but structured compression

### **PID-Net Connection: ⭐⭐⭐⭐ HIGH**

Infini-attention's compressive memory IS the I-stream we need:
```
I-stream = compressive memory that accumulates structured summaries of past context
P-stream = current-window processing
D-stream = difference between current attention pattern and compressed memory expectation

Gate = how much to rely on current (P) vs accumulated (I) vs change signal (D)
```

The key insight: the memory matrix M in Infini-attention uses **linear attention update** (M += k^T · v), which is exactly the associative memory structure from Hopfield networks. Everything converges.

---

## 10. Byte Latent Transformer (BLT) — Dynamic Tokenization

### Paper: Meta, "Byte Latent Transformer" — 2024

### What It Is
Instead of fixed tokenization (BPE), BLT dynamically groups bytes into patches based on **entropy**:
- High-entropy regions (complex text) → small patches (more processing per byte)
- Low-entropy regions (predictable text) → large patches (less processing)

```
Entropy-based patching:
  "The" → one patch (predictable, compress)
  "syzygy" → many small patches (rare, need detail)
```

### Why It's Revolutionary
- **Eliminates tokenizer** — raw bytes in, text out. No BPE artifacts.
- **Adaptive compute** — naturally spends more compute on harder parts
- **7x compute reduction** vs byte-level processing at same quality
- **No vocabulary mismatch** — works across ALL languages and modalities

### **PID-Net Connection: ⭐⭐⭐⭐ HIGH**

PID-Net already processes bytes. BLT's entropy-based patching maps to D-gate:
```
D-gate magnitude = local entropy estimate
High D → small patches (need more processing)
Low D → large patches (compress)
```

The D-stream IS the entropy estimator that BLT needs. If we let D-gate control patch sizes dynamically, we get BLT-style adaptive tokenization for free.

---

## 11. Griffin/Hawk — Google's Linear Recurrence

### Paper: De et al., "Griffin: Mixing Gated Linear Recurrences with Local Attention" — Google DeepMind, 2024

### What It Is
Griffin combines:
1. **Gated Linear Recurrence (GLR):** Simplified SSM with real-valued diagonal state
2. **Local sliding-window attention:** Standard attention but only over nearby tokens

```
GLR:  h_t = a_t ⊙ h_{t-1} + (1-a_t) ⊙ x_t    # gated linear recurrence
      y_t = h_t                                   # output = state
      
Where a_t = σ(W_a · x_t)  # input-dependent gate
```

### Why It Matters
- **Matches Llama-2** at every scale tested (up to 14B)
- **O(T) recurrence + O(local²) attention** — near-linear total
- **Simpler than Mamba** — no custom CUDA kernels needed
- **The gate `a_t` is a forgetting gate** — like LSTM's forget gate

### **PID-Net Connection: ⭐⭐⭐⭐ HIGH**

Griffin's GLR IS a single-stream PID without the decomposition:
```
h_t = a_t · h_{t-1} + (1-a_t) · x_t

This is:
  I-component: a_t · h_{t-1}         (accumulated state, decayed)
  P-component: (1-a_t) · x_t         (current input)
  
Missing: D-component (no change detection)
```

**Griffin is PID-Net minus the D-stream.** Adding a derivative signal to Griffin would give it:
- Surprise detection
- Adaptive forgetting (high D → forget more, reset state)
- Stagnation prevention

---

## 12. MEGA — Moving Average Equipped Gated Attention

### Paper: Ma et al., "Mega: Moving Average Equipped Gated Attention" — Meta, 2023

### What It Is
MEGA combines:
1. **Exponential Moving Average (EMA):** Captures local/sequential patterns
2. **Single-head gated attention:** Captures long-range dependencies
3. **Gated residual:** Controls information flow

```
EMA:    h_t = α · h_{t-1} + (1-α) · x_t          # moving average (local context)
Attn:   a_t = attention(x_t, X)                    # global context
Gate:   g_t = σ(W · [h_t, a_t])                    # combine gate
Output: y_t = g_t ⊙ h_t + (1-g_t) ⊙ a_t          # gated combination
```

### **PID-Net Connection: ⭐⭐⭐⭐ VERY HIGH**

MEGA is literally a P+I system with attention:
```
I-stream = EMA (moving average)
P-stream = current input (through attention)
Gate = how much to rely on each

MISSING: D-stream
```

Again, no derivative/change detection. Adding D to MEGA would give it a signal for when the EMA is stale (high derivative → input changed significantly → EMA is outdated → lower I-gate → rely more on current P).

---

## 13. RetNet — Retentive Networks

### Paper: Sun et al., "Retentive Network" — Microsoft, 2023

### What It Is
RetNet unifies three computation paradigms:
1. **Parallel** (like transformer, for training)
2. **Recurrent** (like RNN, for inference)
3. **Chunkwise** (hybrid, for long sequences)

The retention mechanism:
```
Retention(X) = (Q · K^T ⊙ D) · V

Where D_ij = { γ^(i-j)  if i ≥ j
             { 0         otherwise

γ = exponential decay factor
```

This is causal attention with **exponential decay** — farther tokens get exponentially less weight.

### Why It Matters
- **3x inference speed** vs transformers (recurrent formulation)
- **Same quality** as transformers on language modeling
- **The decay γ IS an I-stream parameter** — it controls memory persistence

### **PID-Net Connection: ⭐⭐⭐⭐ HIGH**

RetNet's exponential decay IS our I-stream's λ:
```
RetNet:  weight ∝ γ^(i-j)
PID I:   I_t = λ · I_{t-1} + (1-λ) · x_t

These are the same mechanism: exponential weighting of the past.
```

But RetNet uses a FIXED γ per head. PID-Net should use **adaptive γ** (controlled by D-gate). Some heads need long memory (small γ, slow decay), others need short memory (large γ, fast decay). And this should change dynamically based on input.

---

## 14. Mixture of Depths — Adaptive Computation Per Token

### Paper: Raposo et al., "Mixture-of-Depths" — Google DeepMind, 2024

### What It Is
Not every token needs the same amount of processing. Easy tokens (common words, predictable sequences) can skip layers. Hard tokens (rare words, surprising context) need full processing.

```
For each token at each layer:
  router_score = W_router · x_t
  if router_score > threshold:
    x_t = full_layer(x_t)      # process normally
  else:
    x_t = x_t                  # skip (residual only)
```

### Why It's Revolutionary
- **50% compute reduction** with NO quality loss
- **Adaptive:** Model learns which tokens need attention
- **Complementary with MoE:** MoE selects WHICH expert; MoD selects WHETHER to process

### **PID-Net Connection: ⭐⭐⭐⭐⭐ NATURAL FIT**

**The D-gate IS the router for Mixture of Depths.**

```
D-gate magnitude → how much processing this token needs
High D (surprising/novel) → full processing (all PID streams active)
Low D (predictable) → skip (residual only, minimal compute)
```

This is exactly what PID-Net's adaptive compute was designed for. The gate values determine compute allocation. We don't need a separate router — the D-stream IS the router.

---

## 15. Memory Transformer / Memorizing Transformers

### Paper: Wu et al., "Memorizing Transformers" — Google, 2022

### What It Is
Standard transformers forget everything outside their context window. Memorizing Transformers add a **non-differentiable external memory** (kNN lookup) that stores key-value pairs from past sequences.

```
1. Normal attention within context window
2. kNN search over external memory bank (millions of stored KV pairs)
3. Combine: output = local_attention + λ * memory_attention
```

### Why It Matters
- **Non-parametric memory:** Memory capacity scales with storage, not parameters
- **Exact retrieval:** kNN gives precise recall (no compression loss)
- **Perplexity drops significantly** — the model "remembers" rare patterns it saw before

### **PID-Net Connection: ⭐⭐⭐⭐ HIGH**

The I-stream could be a **hybrid**: parametric (fast, learned) + non-parametric (exact, stored).

```
I_stream = λ_param · parametric_memory(x_t) + λ_nonparam · kNN_memory(x_t)

Where:
  parametric_memory = compressed/learned representation (fast but lossy)
  kNN_memory = exact stored key-value pairs (slow but precise)
  λ values controlled by D-gate (high surprise → use exact memory, low surprise → use fast compressed)
```

---

## 16. Associative Memory Transformers

### Paper: Schlag et al., "Linear Transformers Are Secretly Fast Weight Programmers" — 2021

### What It Is
This paper proves that **linear attention is equivalent to a fast weight programmer** — a system that writes to and reads from a weight matrix.

```
Linear attention: Σ_i φ(q)φ(k_i)^T · v_i
Fast weights:     M = Σ_i k_i · v_i^T    (write)
                  y = M · q                (read)

These are IDENTICAL.
```

This means linear attention IS associative memory. The "attention matrix" is a learned associative memory that stores key-value associations.

### **PID-Net Connection: ⭐⭐⭐⭐⭐ UNIFYING**

This closes the loop:
```
Attention = Hopfield Memory = Fast Weight Programming = Associative Memory

Our I-stream should be a fast weight programmer:
  M_t = λ · M_{t-1} + η · k_t · v_t^T    (write with decay)
  read_t = M_t · q_t                       (associative retrieval)
  
This is:
  - EMA of outer products (structured, not lossy averaging)
  - Linear attention (efficient, O(T))
  - Hopfield update (energy-based, convergent)
  - Fast weight programming (learned associations)

ALL THE SAME THING, just different perspectives.
```

---

## 17. Cognitive Architectures — ACT-R, SOAR, Global Workspace

### What They Are
Before deep learning, AI researchers built **cognitive architectures** — complete models of human cognition based on psychology experiments.

**ACT-R (Anderson, 1993-present):**
- Declarative memory (facts) — I-stream
- Procedural memory (skills/rules) — learned gate patterns
- Goal buffer (current objective) — P-stream
- Perceptual buffer (current input) — P-stream
- Conflict resolution (which rule to fire) — gate mechanism
- **Base-level activation:** memories decay with power law, not exponential

**Global Workspace Theory (Baars, 1988):**
- Specialized unconscious modules process information in parallel
- A "global workspace" broadcasts selected information to all modules
- Consciousness = what's in the workspace
- Modules compete for workspace access

**SOAR (Laird, Newell, 1987):**
- Working memory (current state) — P-stream
- Long-term memory (procedural + semantic + episodic) — I-stream variants
- Decision cycle: propose → evaluate → apply
- Impasse detection → learning (chunking) — D-stream (detecting when processing fails)

### **PID-Net Connection: ⭐⭐⭐⭐⭐ FOUNDATIONAL**

These architectures spent DECADES studying how cognition works. Key insights for us:

1. **Memory decay is POWER LAW, not exponential** (ACT-R):
```
Activation(t) = ln(Σ t_j^(-d))    # d ≈ 0.5, power law

vs our current:
I_t = λ^t · I_0                    # exponential decay
```
Power law = slow initial decay, then gradual fade. Exponential = fast initial decay. **Power law matches human memory curves.** We should use power-law decay for I-stream.

2. **Multiple memory types** (ACT-R/SOAR):
- Declarative (facts) → associative I-stream
- Procedural (how-to) → learned gate patterns (NOT stored in I)
- Episodic (sequences of events) → temporal I-stream with ordering
We have ONE I-stream. We need at least two: **associative** (what) and **episodic** (when/order).

3. **Global Workspace broadcasting** (Baars):
- Selected information gets broadcast to ALL processing modules
- This is what the gate does — it selects which PID stream to broadcast
- Extension: gate should broadcast to MULTIPLE downstream layers, not just output

4. **Impasse detection = D-gate** (SOAR):
- SOAR detects when it can't make progress (impasse)
- This triggers a learning episode (chunking)
- D-gate high magnitude = impasse = trigger memory write + strategy change

---

## 18. Power Law Decay — The Right Forgetting Curve

### From ACT-R + Cognitive Psychology

Human memory follows a **power law**, not exponential:
```
Recall probability ∝ t^(-β)    where β ≈ 0.5

vs exponential:
Recall probability ∝ e^(-λt)
```

**Why this matters:** Power law decay means:
- Recent memories fade slowly (high retention in short term)
- Old memories fade very slowly (long tail — you remember childhood events)
- Exponential decay is too aggressive — kills useful memories too fast

**Implementation:**
```python
# Instead of:
I_t = λ · I_{t-1} + (1-λ) · x_t          # exponential decay

# Use:
I_t = Σ_{j=0}^{t} (t - j + 1)^(-β) · x_j  # power law weighted sum

# Efficient approximation:
# Maintain memories at multiple timescales (1, 2, 4, 8, 16, 32, ...)
# Weight by power law over timescales
```

This connects to **multi-timescale PID banks**. Instead of arbitrary timescales, use power-law-spaced timescales.

---

## SYNTHESIS: The Convergence Map

Every single architecture we've studied converges to the same components:

```
┌─────────────────────────────────────────────────────┐
│              UNIVERSAL INTELLIGENCE COMPONENTS       │
├─────────────────────────────────────────────────────┤
│                                                      │
│  P-STREAM (Perception/Current)                      │
│  ├── Standard input processing                       │
│  ├── = Griffin's (1-a_t)·x_t                        │
│  ├── = MEGA's attention output                       │
│  ├── = ACT-R's perceptual buffer                    │
│  └── = Predictive coding's bottom-up signal         │
│                                                      │
│  I-STREAM (Memory/Accumulated)                      │
│  ├── Associative memory (Hopfield/fast weights)     │
│  ├── = Linear attention's weight matrix              │
│  ├── = RetNet's exponential decay                    │
│  ├── = Griffin's a_t · h_{t-1}                      │
│  ├── = MEGA's EMA                                    │
│  ├── = Infini-attention's compressive memory         │
│  ├── = ACT-R's declarative memory                   │
│  ├── = xLSTM's matrix memory                        │
│  └── Should use POWER LAW decay, not exponential    │
│                                                      │
│  D-STREAM (Change/Prediction Error)                 │
│  ├── Prediction error (predictive coding)           │
│  ├── = Differential Transformer's attention diff     │
│  ├── = SOAR's impasse detection                     │
│  ├── = Titans' surprise signal                      │
│  ├── = RL's TD error                                │
│  ├── Controls: memory write strength (D→I coupling) │
│  ├── Controls: compute depth (Mixture of Depths)    │
│  ├── Controls: patch size (BLT-style)               │
│  └── Controls: I-decay rate (adaptive forgetting)   │
│                                                      │
│  GATE (Selection/Consciousness)                     │
│  ├── Precision weighting (Free Energy Principle)    │
│  ├── = Global Workspace selection                    │
│  ├── = Consciousness Prior bottleneck               │
│  ├── = ACT-R's conflict resolution                  │
│  ├── = MoE routing                                   │
│  └── = MoD skip decision                            │
│                                                      │
│  META-LEARNING (TTT)                                │
│  ├── Weights update during inference                 │
│  ├── = Brain's online learning                       │
│  ├── = Dopamine-modulated plasticity                │
│  └── Learning rate controlled by D-stream           │
│                                                      │
└─────────────────────────────────────────────────────┘
```

---

## THE 20 KEY PAPERS TO READ (ordered by PID-Net relevance)

1. **Ramsauer — "Hopfield Networks is All You Need"** (2020) — attention = memory
2. **Sun — "Learning to (Learn at Test Time)"** (2024) — TTT layers
3. **Friston — "The Free Energy Principle"** (2010) — the grand theory
4. **Rao & Ballard — "Predictive Coding in Visual Cortex"** (1999) — PID IS prediction
5. **Ye — "Differential Transformer"** (2024) — D-stream for attention
6. **Munkhdalai — "Leave No Context Behind (Infini-Attention)"** (2024) — compressive memory
7. **Schlag — "Linear Transformers Are Secretly Fast Weight Programmers"** (2021) — I-stream theory
8. **Graves — "Neural Turing Machines"** (2014) — explicit memory management
9. **Bengio — "The Consciousness Prior"** (2017) — gate = consciousness
10. **Beck — "xLSTM"** (2024) — matrix memory
11. **Kanerva — "Sparse Distributed Memory"** (1988) — biological memory model
12. **Raposo — "Mixture-of-Depths"** (2024) — D-gate as compute router
13. **De — "Griffin"** (2024) — PID-without-D
14. **Ma — "MEGA"** (2023) — PID-without-D (different variant)
15. **Sun — "RetNet"** (2023) — exponential decay attention
16. **Anderson — "ACT-R"** (1993-2024) — cognitive architecture, power-law decay
17. **Millidge — "Predictive Coding Approximates Backprop"** (2022) — no backprop needed?
18. **Greydanus — "Hamiltonian Neural Networks"** (2019) — energy conservation
19. **Behrouz — "Titans"** (2025) — surprise-gated memory
20. **Hasani — "Liquid Time-Constant Networks"** (2021) — adaptive dynamics

---

*"Every architecture in AI history is a partial implementation of the same universal pattern: Perceive, Remember, Detect Change, Select. PID-Net is the first to implement all four explicitly."*

— Ion, March 2026
