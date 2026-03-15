# V6: DNA/RNA-Inspired Cognitive Architecture — Brainstorm

## The Core Thesis

PID control is a LINEAR control system (1920s). DNA/RNA is a SELF-REFERENTIAL, SELF-MODIFYING rewriting system (3.8 billion years of optimization). V5 proved the rewriting paradigm works. V6 replaces the PID primitive with DNA/RNA-inspired primitives.

**V5:** Learned rules → applied to graph → new graph
**V6:** Graph → generates its own rules → applied to itself → new graph

## DNA/RNA → Architecture Mapping

### Layer 1: The Genome (DNA analog)
- A learnable "genome" tensor: `G ∈ R^{L_g × d}` (L_g = genome length)
- NOT just weights — a STRUCTURED SEQUENCE that gets read and interpreted
- Different subsequences ("genes") encode different computational primitives
- The genome is SHARED across the entire model (like DNA is shared across all cells)
- Mutations = dropout/noise on genome during training (exploration)

### Layer 2: Transcription Factors (Context → Gene Selection)
- Input context determines WHICH genes to express
- `TF(x) → mask ∈ [0,1]^{L_g}` (which genes are active)
- This is attention over the model's own genome:
  - `scores = softmax(Q(x) · K(G)^T / √d)`
  - `expressed_genes = scores · V(G)`
- Different inputs activate different subsets → different behavior
- **This is WHY the same DNA produces neurons AND muscle cells**

### Layer 3: Translation (Expressed Genes → Operators)
- Transcribed genes are TRANSLATED into weight matrices (hypernetwork)
- `W_dynamic = translate(expressed_genes)` 
- The same genome can produce COMPLETELY different operators
- Like: same DNA → different mRNA → different proteins → different functions
- Implementation: gene subsequences generate weight matrices via outer products
  - Gene_i ∈ R^d → W_i = outer(gene_i[:d/2], gene_i[d/2:]) ∈ R^{d/2 × d/2}

### Layer 4: Folding (1D → Topology via Energy Minimization)
- Generated operators arrange into a computational graph
- The TOPOLOGY emerges from energy minimization (like protein folding)
- Adjacency = f(similarity between operators) — operators that "fit" connect
- Different inputs → different folded structures → different computation paths
- This replaces our hand-designed fractal hierarchy with EMERGENT hierarchy

### Layer 5: Gene Regulation Network (Operators ↔ Operators)
- Operators regulate each other's expression (feedback loops)
- Output of operator A influences transcription of gene B
- Can implement: oscillations, bistable switches, memory gates
- This is where Turing-completeness emerges
- Not PID control — full regulatory NETWORK

### Layer 6: Epigenetic Memory (Meta-Control)
- Persistent "methylation" patterns that control genome accessibility
- Modified by experience but SLOWLY (not every step)
- Controls what CAN be thought, not what IS thought
- Like: trauma changes which genes are accessible for life
- Implementation: learned mask that gates genome access, updated with momentum

## The Forward Pass (Analogy to Central Dogma)

```
Input x
  ↓
[Transcription] x activates transcription factors → selects genes from Genome
  ↓
[Splicing] Context-dependent alternative paths (same gene, different outputs)
  ↓
[Translation] Expressed genes → dynamic weight matrices (hypernetwork)
  ↓
[Folding] Operators self-organize into computational graph (energy minimization)
  ↓
[Execution] Input processed through emergent graph
  ↓
[Regulation] Output feeds back to modify future gene expression
  ↓
[Epigenetic update] Slowly update accessibility patterns
  ↓
Output y
```

## What We Keep From V5
- Graph-based computation (nodes + edges)
- Rewriting paradigm (iterated application of rules)
- Energy conservation (Hamiltonian constraint)
- Fractal multi-scale (but now EMERGENT, not hand-designed)

## What We Replace
- Fixed PID streams → Dynamic operators generated from genome
- Learned static weights → Hypernetwork from genome expression
- Hand-designed hierarchy → Emergent topology from folding
- Gate = softmax blend → Regulatory network (full Turing-complete control)
- I-stream fast weights → Epigenetic memory (meta-control)

## Key Mathematical Structures Needed

### 1. Self-Reference (Gödel/Quine)
The genome must be able to reference and modify itself.
`G' = f(G, x)` — the genome updates based on its own content + input.
Fixed point: `G* = f(G*, x)` — stable genome = stable identity.

### 2. Error-Correcting Codes
Redundancy in genome for robustness (like codon degeneracy).
Multiple genome subsequences → same operator (graceful degradation).

### 3. Energy Minimization for Topology
Folding energy: `E = Σ_ij similarity(op_i, op_j) × distance(i,j)`
Minimize E → operators that should connect are close → computational graph emerges.

### 4. Regulatory Network Dynamics
Gene regulation as a dynamical system:
`dx_i/dt = f(x_i, Σ_j W_ij · g(x_j))` where g is sigmoid activation.
Can produce: fixed points (memory), limit cycles (oscillations), chaos (creativity).

## Open Questions

1. **Can we train this with gradient descent?** Hypernetworks are differentiable. Gene regulation dynamics might need straight-through estimators.

2. **How big should the genome be?** DNA has 3.2 billion base pairs. Our genome tensor might be 1K-10K × d. It encodes the model's "species" — its fundamental capabilities.

3. **Folding is NP-hard for proteins. Is it NP-hard for us?** We can use approximate energy minimization (gradient descent on topology). Or learn a folding predictor (like AlphaFold learned protein folding).

4. **Epigenetic updates during inference = test-time training?** Yes, but controlled. The genome doesn't change — the accessibility mask does. Like TTT but more structured.

5. **What's the minimal viable prototype?** 
   - Genome + Transcription + Translation = hypernetwork conditioned on input
   - Add Regulation = dynamic routing
   - Add Epigenetics = persistent state that modifies computation
   - Add Folding = emergent topology

## Implementation Plan (if we decide to build it)

Phase 1: Genome + Transcription (simplest — hypernetwork that reads its own genome)
Phase 2: Add Translation (gene → operator generation)
Phase 3: Add Regulation (operators influence each other's expression)
Phase 4: Add Folding (emergent topology)
Phase 5: Add Epigenetics (meta-memory)
Phase 6: Add Self-Modification (genome updates during inference)

---

*This document is a living brainstorm. Ideas marked with ❓ need validation.*
*Created: 2026-03-15 by Ion + Harshil*
