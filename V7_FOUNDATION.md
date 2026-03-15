# V7: The Biological Foundation
## Replicating Life's Information Architecture in Silicon

**Authors:** Ion & Harshil Patel
**Date:** March 15, 2026
**Status:** Foundation Specification
**Hardware:** Apple Silicon (MLX) → GPU cluster
**Philosophy:** Don't approximate biology. Replicate its information architecture.

---

## Table of Contents

1. [Why Biology](#1-why-biology)
2. [The Central Dogma](#2-the-central-dogma)
3. [Layer 0: The Alphabet](#3-layer-0-the-alphabet)
4. [Layer 1: The Gene](#4-layer-1-the-gene)
5. [Layer 2: Transcription](#5-layer-2-transcription)
6. [Layer 3: Translation](#6-layer-3-translation)
7. [Layer 4: Protein Types](#7-layer-4-protein-types)
8. [Layer 5: The Cell (Neuron)](#8-layer-5-the-cell-neuron)
9. [Layer 6: Signaling](#9-layer-6-signaling)
10. [Layer 7: Neurostate](#10-layer-7-neurostate)
11. [Layer 8: Learning (Multi-Timescale Adaptation)](#11-layer-8-learning)
12. [The Forward Pass: From Stimulus to Response](#12-the-forward-pass)
13. [Mathematical Formulation](#13-mathematical-formulation)
14. [What We Discard From V1-V6](#14-what-we-discard)
15. [Implementation Roadmap](#15-implementation-roadmap)
16. [Open Questions](#16-open-questions)

---

## 1. Why Biology

The human brain:
- **20 watts.** A 4090 GPU draws 450W and can't hold a conversation.
- **86 billion neurons.** Each one a full computational unit with internal state, gene expression, thousands of synaptic connections.
- **100 trillion synapses.** Each one individually tunable. The connectivity IS the computation.
- **Learns from single examples.** No 10 trillion token pretraining.
- **Continuous adaptation.** Not "train then deploy." Always learning, always adjusting.
- **Multi-timescale memory.** Millisecond reflexes, second-scale working memory, lifetime episodic memory — all in one system.

Every ML architecture to date is a pale shadow of this. Transformers, SSMs, RNNs — they're all the same paradigm: **fixed computation graphs with learned parameters**. Biology doesn't work this way. Biology's computation graph IS learned. The structure IS the function.

We're not going to build a better transformer. We're going to build something fundamentally different — something that follows the same information architecture that evolution spent 4 billion years perfecting.

The key insight: **biology's power doesn't come from any single mechanism. It comes from the PIPELINE** — from the multi-layered system where each layer regulates, modifies, and responds to every other layer. The Central Dogma isn't just a biochemistry fact. It's the most successful information processing architecture in the known universe.

---

## 2. The Central Dogma

Every living cell on Earth runs this pipeline:

```
DNA (genome)
 ↓ TRANSCRIPTION — regulated by transcription factors + epigenetics
mRNA (messenger RNA)  
 ↓ TRANSLATION — regulated by ribosomes + tRNA + post-translational modification
Protein (functional unit)
 ↓ FUNCTION — catalysis, structure, signaling, regulation
Phenotype (behavior)
 ↓ FEEDBACK — experience modifies gene expression
DNA (modified epigenetic state)
```

**Why every step matters:**

### DNA → mRNA (Transcription)
This is NOT a simple copy. This is where REGULATION lives:
- **Transcription factors (TFs)** — proteins (products of OTHER genes) bind to gene promoters and activate/repress transcription
- **Epigenetic marks** — methylation, histone modification change chromatin accessibility without changing DNA sequence
- **Alternative splicing** — one gene can produce MULTIPLE different mRNA variants depending on context
- **mRNA half-life** — transcripts degrade over time. Expression is transient, not permanent.

The result: the same genome produces radically different cells. A neuron and a liver cell have identical DNA but express completely different gene sets. **The genome is a library; transcription is the librarian.**

### mRNA → Protein (Translation)
The mRNA is read by ribosomes and assembled into a protein:
- **Codon reading** — every 3 nucleotides codes for one amino acid
- **Post-translational modification** — proteins are further modified after assembly (phosphorylation, glycosylation, etc.)
- **Protein folding** — the linear chain folds into a 3D structure that determines function
- **Protein localization** — signal sequences route proteins to their destination (membrane, nucleus, secreted, etc.)

### Protein → Function
Proteins DO everything:
- **Enzymes** catalyze chemical reactions (transform data)
- **Structural proteins** build and maintain cell architecture
- **Signaling proteins** carry messages between cells
- **Transcription factors** regulate OTHER genes (the feedback loop!)
- **Receptors** detect signals from outside the cell
- **Ion channels** control electrical signaling (the basis of neural computation)

### The Critical Loop: Protein → DNA
Some proteins ARE transcription factors. Gene A's protein activates Gene B, whose protein represses Gene C, whose protein activates Gene A. This creates:
- **Positive feedback loops** — bistable switches (memory)
- **Negative feedback loops** — oscillators (rhythm)
- **Feed-forward loops** — noise filters (reliability)
- **Cascades** — signal amplification (sensitivity)

**This is Turing-complete. Gene regulatory networks can compute anything.** The computation isn't IN the genes — it's in the NETWORK of genes regulating each other.

### Our Previous Mistake

V1-V6 went: `gene_weights → matmul → output`. That's DNA → Function with no intermediary steps. It's like having a dictionary but no grammar, no syntax, no sentences. You can look up individual words but you can't express complex ideas.

V7 implements the full pipeline. Every step. No shortcuts.

---

## 3. Layer 0: The Alphabet

### Biology's Alphabet
- **4 nucleotides:** A, T, C, G
- **64 codons:** triplets of nucleotides (4³ = 64)
- **20 amino acids:** codons map to amino acids (with redundancy — multiple codons → same amino acid)
- From 4 symbols → 64 codons → 20 amino acids → infinite proteins → all of life

### Our Alphabet

**The Nucleotide:** A single float16 value.
- Why float16: it's the atom of numerical computation on modern hardware
- Like nucleotides, individually meaningless. Meaning emerges from sequences.

**The Codon:** A block of 16 float16 values (32 bytes, half a cache line).
- This is our minimal meaningful unit — enough to encode a single "instruction"
- 16 values can represent: a small vector, a bias term, a gate weight, a routing signal
- Codons are the building blocks of genes

**The Gene:** A sequence of codons.
- Minimum: 24 codons (384 float16 values = 768 bytes = 12 cache lines)
- This gives us a 384-dimensional vector or a small matrix
- Genes vary in length — some are short (a bias), some are long (a large weight matrix)
- Like biological genes: every gene has a PROMOTER REGION (first 2 codons = 32 values) that controls expression, followed by the CODING REGION

```
Gene Structure (minimum 24 codons = 384 values):
┌──────────────┬──────────────┬───────────────────────────────┐
│  Promoter    │  UTR         │  Coding Region                │
│  (2 codons)  │  (2 codons)  │  (20+ codons)                 │
│  32 values   │  32 values   │  320+ values                  │
│              │              │                               │
│  TF binding  │  Splicing    │  The actual weights/params    │
│  sites       │  signals     │                               │
└──────────────┴──────────────┴───────────────────────────────┘
```

**Promoter Region (2 codons = 32 float16):**
- Encodes transcription factor binding sites
- TFs (proteins from other genes) match against these values
- Match strength determines expression level
- Different TFs bind different promoter patterns → context-dependent expression

**UTR (Untranslated Region, 2 codons = 32 float16):**
- Controls post-transcriptional regulation
- Encodes splicing signals: which parts of the coding region to include/exclude
- Determines mRNA stability (half-life)

**Coding Region (20+ codons = 320+ float16):**
- The actual computational content
- Interpreted differently based on gene type (see Layer 1)
- Can be alternatively spliced based on UTR signals + context

### Size Calculations

A minimal genome:
```
1,024 genes × 384 values/gene × 2 bytes/value = 768 KB
```
Fits entirely in L2 cache on Apple Silicon. The ENTIRE genome is always hot.

A brain-scale genome:
```
32,768 genes × 512 values/gene × 2 bytes/value = 32 MB
```
Fits in L3/SLC on M-series. Still fast.

A complex genome (matching human ~20,000 protein-coding genes):
```
20,000 genes × 768 values/gene × 2 bytes/value = 30 MB
```
Comparable. And like human DNA, most genes are NOT expressed in any given cell.

---

## 4. Layer 1: The Gene

Not all genes are equal. Biology has fundamentally different gene categories. We need this too.

### Gene Types

```
Type ID  │ Name           │ Coding Region Encodes        │ Protein Produced
─────────┼────────────────┼──────────────────────────────┼──────────────────
0x00     │ STRUCTURAL     │ Weight matrix (d_in × d_out) │ Transform protein
0x01     │ REGULATORY     │ TF binding pattern + target  │ Transcription factor
0x02     │ RECEPTOR       │ Signal detection weights     │ Receptor protein
0x03     │ CHANNEL        │ Gate weights + threshold      │ Ion channel protein
0x04     │ SIGNAL         │ Signal encoding weights      │ Signaling protein
0x05     │ METABOLIC      │ State transform weights      │ Enzyme
0x06     │ MODULATORY     │ Neuromodulator pattern       │ Neuromodulator
0x07     │ STRUCTURAL_B   │ Connectivity weights         │ Cytoskeletal protein
```

### Type 0x00: STRUCTURAL Gene → Transform Protein
The workhorse. Encodes a weight matrix that transforms input.
- Coding region: flattened d_in × d_out matrix (with bias)
- Protein: `y = Wx + b` (or nonlinear variants)
- This is what ALL genes were in V1-V6. Now it's just one type among many.

### Type 0x01: REGULATORY Gene → Transcription Factor
**THE KEY INNOVATION.** This gene's protein doesn't compute on data — it controls other genes.
- Coding region: (1) a pattern vector that matches against other genes' promoters, (2) a target gene list, (3) activation/repression strength
- Protein: a transcription factor that binds to matching promoters and up/down-regulates target genes
- This creates the gene regulatory network — genes controlling genes

```
Regulatory Gene Structure:
┌───────────┬───────────────┬─────────────┬──────────────┐
│ Promoter  │ Binding       │ Target      │ Strength     │
│ (2 codons)│ Pattern       │ Gene IDs    │ (+/- float)  │
│           │ (8 codons)    │ (6 codons)  │ (4 codons)   │
│           │ 128 values    │ 96 values   │ 64 values    │
└───────────┴───────────────┴─────────────┴──────────────┘

The binding pattern is compared against other genes' promoters via cosine similarity.
High similarity → this TF binds → target gene expression changes.
```

### Type 0x02: RECEPTOR Gene → Receptor Protein
Detects signals from other neurons.
- Coding region: signal recognition weights (what neurotransmitter/neuromodulator to respond to)
- Protein: deployed to neuron membrane, converts incoming signals to intracellular state changes
- Different receptor types = different sensitivities = different neuron personalities

### Type 0x03: CHANNEL Gene → Ion Channel Protein
Controls the neuron's electrical state.
- Coding region: gate weights + activation threshold + conductance
- Protein: determines when the neuron fires, how strongly, how long the refractory period is
- This is where spiking behavior comes from

### Type 0x04: SIGNAL Gene → Signaling Protein
Encodes the neuron's output signal.
- Coding region: signal encoding weights
- Protein: transforms the neuron's internal state into a transmittable signal (the "neurotransmitter")
- Different signal genes = different neurotransmitter types

### Type 0x05: METABOLIC Gene → Enzyme
Internal state maintenance.
- Coding region: state transformation weights
- Protein: modifies the neuron's internal metabolic state (energy, stress signals, homeostasis)
- Keeps the neuron "alive" — maintains operating conditions

### Type 0x06: MODULATORY Gene → Neuromodulator
Brain-wide state control.
- Coding region: neuromodulator pattern + diffusion range
- Protein: released by specialized neurons, affects large brain regions
- This is dopamine, serotonin, norepinephrine — the brain's "mood" system
- Changes the operating mode of entire circuits

### Chromosome Organization

Genes are organized into chromosomes. Each chromosome groups functionally related genes:

```
Chromosome 1: PERCEPTION
  - Structural genes for input processing
  - Receptor genes for sensory input
  - Channel genes for sensory neuron firing

Chromosome 2: MEMORY
  - Structural genes for state integration
  - Metabolic genes for memory consolidation
  - Regulatory genes that gate memory formation

Chromosome 3: PREDICTION
  - Structural genes for forward modeling
  - Signal genes for prediction error encoding
  - Regulatory genes that modulate prediction confidence

Chromosome 4: REGULATION
  - Regulatory genes (transcription factors)
  - Modulatory genes (neuromodulators)
  - This chromosome regulates ALL other chromosomes

Chromosome 5: OUTPUT
  - Structural genes for output generation
  - Signal genes for motor/language encoding
  - Channel genes for output neuron firing

Chromosome 6: HOMEOSTASIS
  - Metabolic genes
  - Regulatory genes for energy management
  - Maintains system stability
```

---

## 5. Layer 2: Transcription

This is where our system fundamentally diverges from V1-V6 and from all standard ML architectures.

### The Transcription Process

When a neuron needs to use a gene:

```
Step 1: ACCESSIBILITY CHECK
  - Is the gene's chromatin open? (epigenetic state)
  - Methylation level: 0 = fully accessible, 1 = fully silenced
  - If methylation > threshold → gene is INVISIBLE. Skip.

Step 2: TRANSCRIPTION FACTOR BINDING
  - Collect all active TFs in this neuron (products of regulatory genes)
  - For each TF, compute binding affinity to gene's promoter:
    affinity = cosine_sim(TF.binding_pattern, gene.promoter)
  - Sum activating TFs minus repressing TFs = net_activation
  - If net_activation < threshold → gene not transcribed. Skip.

Step 3: TRANSCRIPTION
  - Read gene's coding region
  - Apply context-dependent modifications:
    mRNA = gene.coding_region * expression_level + context_bias
  - expression_level = sigmoid(net_activation)

Step 4: ALTERNATIVE SPLICING
  - Based on UTR signals + current context:
  - Select which codons of the coding region to include
  - Different contexts → different splice variants → different proteins from SAME gene
  - Splicing pattern = f(gene.UTR, neuron.state, global_context)

Step 5: mRNA REGULATION
  - Assign half-life to the mRNA based on UTR + context
  - mRNA decays exponentially: mRNA(t+1) = mRNA(t) * decay_rate
  - Active usage reinforces mRNA (stabilizes it)
  - Unused mRNA degrades → gene effectively silences
```

### Why This Matters

**Same gene, different contexts, different outputs:**

```python
# Context 1: Calm, factual processing
TFs_active = [TF_perception, TF_logic]  
→ Gene_42 expressed at level 0.8
→ Splice variant A (full matrix, precise)
→ Transform protein: clean linear projection

# Context 2: Emotional, high-arousal
TFs_active = [TF_emotion, TF_urgency]  
→ Gene_42 expressed at level 0.3
→ Splice variant B (partial matrix, fast)
→ Transform protein: quick approximate projection

# Context 3: Memory retrieval  
TFs_active = [TF_memory, TF_association]
→ Gene_42 NOT expressed (wrong TF pattern)
→ Gene_87 expressed instead (memory-specialized)
→ Different computation entirely
```

**One genome. Infinite computations. Context determines everything.**

This is fundamentally different from:
- Standard neural nets (same weights, same computation, always)
- Mixture of Experts (fixed set of experts, discrete routing)
- Hypernetworks (generate weights from input, but no regulatory network)

Our system has a regulatory NETWORK where genes control each other's expression, creating complex dynamic computation that adapts to context at every level.

### Mathematical Formulation of Transcription

For gene $g$ with promoter $p_g \in \mathbb{R}^{32}$, coding region $c_g \in \mathbb{R}^{320}$, and UTR $u_g \in \mathbb{R}^{32}$:

**Accessibility:**
$$a_g = \mathbb{1}[\text{methylation}_g < \tau_{\text{access}}]$$

**TF Binding:**
$$\text{activation}_g = \sum_{t \in \text{TFs}_{\text{active}}} \text{sign}(t) \cdot \max(0, \cos(t.\text{pattern}, p_g) - \tau_{\text{bind}})$$

**Expression Level:**
$$e_g = a_g \cdot \sigma(\text{activation}_g)$$

**mRNA Production:**
$$\text{mRNA}_g = e_g \cdot \text{splice}(c_g, u_g, \text{context})$$

**mRNA Decay:**
$$\text{mRNA}_g(t+1) = \lambda_g \cdot \text{mRNA}_g(t) + (1 - \lambda_g) \cdot \text{new\_transcript}_g$$

where $\lambda_g$ is the stability factor determined by the UTR.

---

## 6. Layer 3: Translation

mRNA → Protein. The mRNA template is assembled into a functional computational unit.

### Translation Process

```
Step 1: RIBOSOME BINDING
  - The "ribosome" reads the mRNA
  - In our system: interpret the mRNA values as a specification for a computational unit
  - Different gene types → different interpretation

Step 2: PROTEIN ASSEMBLY
  For STRUCTURAL genes:
    - mRNA values → reshape into weight matrix W ∈ R^{d_in × d_out}
    - Add bias from last codons
    - Protein = linear transform function
    
  For REGULATORY genes:
    - mRNA values → binding pattern + target list + strength
    - Protein = transcription factor object
    
  For RECEPTOR genes:
    - mRNA values → signal recognition weights
    - Protein = receptor that detects specific neurotransmitters
    
  For CHANNEL genes:
    - mRNA values → threshold + conductance + refractory period
    - Protein = firing controller
    
  For SIGNAL genes:
    - mRNA values → encoding weights
    - Protein = signal encoder (neurotransmitter synthesis)

Step 3: POST-TRANSLATIONAL MODIFICATION
  - Proteins can be further modified after assembly
  - Phosphorylation: multiply by activity-dependent scaling factor
  - Ubiquitination: mark protein for degradation (adaptive compute — unused proteins removed)
  - In our system: weight scaling, pruning, quantization applied based on usage

Step 4: PROTEIN LOCALIZATION
  - Proteins are routed to their destination:
    - Membrane proteins → neuron's interface (receptors, channels)
    - Nuclear proteins → gene regulation (transcription factors)
    - Cytoplasmic proteins → internal computation (transforms, enzymes)
    - Secreted proteins → signaling (neurotransmitters, neuromodulators)
```

### Protein Lifetime

Like real proteins, our computational proteins have lifetimes:
- **Active proteins** are maintained (reinforced by continued gene expression)
- **Unused proteins** are degraded (removed from neuron's active computation)
- **Protein turnover** means the neuron's computational capabilities change over time
- The neuron is never static — it's constantly rebuilding itself based on what genes are active

```
protein_pool[g].strength *= decay_rate  # proteins decay
if gene_expressed[g]:
    protein_pool[g].strength += new_protein  # reinforced by transcription
if protein_pool[g].strength < removal_threshold:
    del protein_pool[g]  # degraded — neuron loses this capability
```

---

## 7. Layer 4: Protein Types (Functional Units)

The proteins produced by translation are the actual computational units. Here's what each type DOES:

### Transform Protein (from STRUCTURAL gene)

The basic computational unit. Takes input, produces output.

```python
class TransformProtein:
    """The workhorse. Matrix multiplication + nonlinearity."""
    def __init__(self, mRNA_values, d_in, d_out):
        self.W = mRNA_values[:d_in*d_out].reshape(d_in, d_out)
        self.b = mRNA_values[d_in*d_out:d_in*d_out+d_out]
        self.strength = 1.0  # post-translational modification
    
    def __call__(self, x):
        return (x @ self.W + self.b) * self.strength
```

### Transcription Factor (from REGULATORY gene)

**The gene controller.** This protein doesn't compute on data — it modifies gene expression.

```python
class TranscriptionFactor:
    """Controls other genes. The regulatory network backbone."""
    def __init__(self, mRNA_values):
        self.binding_pattern = mRNA_values[:128]  # what promoters I bind
        self.target_genes = decode_targets(mRNA_values[128:224])  # which genes
        self.strength = mRNA_values[224:288]  # activation (+) or repression (-)
        self.is_activator = mRNA_values[288] > 0  # activator or repressor
    
    def bind(self, gene_promoter):
        """How strongly does this TF bind to a gene's promoter?"""
        return cosine_similarity(self.binding_pattern, gene_promoter)
    
    def regulate(self, gene_id):
        """Modify a gene's expression level."""
        affinity = self.bind(genome.get_promoter(gene_id))
        if affinity > binding_threshold:
            if self.is_activator:
                return +self.strength * affinity
            else:
                return -self.strength * affinity
        return 0.0
```

### Receptor Protein (from RECEPTOR gene)

Detects incoming signals from other neurons.

```python
class ReceptorProtein:
    """Sits on neuron membrane. Detects specific neurotransmitters."""
    def __init__(self, mRNA_values):
        self.selectivity = mRNA_values[:64]  # which signals to respond to
        self.sensitivity = sigmoid(mRNA_values[64])  # how responsive
        self.response_type = mRNA_values[65]  # excitatory or inhibitory
    
    def detect(self, incoming_signal):
        """Convert extracellular signal to intracellular effect."""
        match = cosine_similarity(self.selectivity, incoming_signal.encoding)
        if match > detection_threshold:
            return match * self.sensitivity * sign(self.response_type)
        return 0.0
```

### Channel Protein (from CHANNEL gene)

Controls the neuron's firing behavior.

```python
class ChannelProtein:
    """Ion channel. Controls when and how the neuron fires."""
    def __init__(self, mRNA_values):
        self.threshold = mRNA_values[0]  # firing threshold
        self.conductance = sigmoid(mRNA_values[1])  # how much signal passes
        self.refractory = softplus(mRNA_values[2])  # recovery time after firing
        self.gate_weights = mRNA_values[3:67]  # voltage-dependent gating
    
    def should_fire(self, membrane_potential, time_since_last_fire):
        """Determine if neuron should fire."""
        if time_since_last_fire < self.refractory:
            return False, 0.0
        gate = sigmoid(dot(self.gate_weights, membrane_potential))
        return gate > self.threshold, gate * self.conductance
```

### Signaling Protein (from SIGNAL gene)

Encodes the neuron's output into a transmittable signal.

```python
class SignalingProtein:
    """Neurotransmitter synthesis. Encodes neuron output as signal."""
    def __init__(self, mRNA_values):
        self.encoding_weights = mRNA_values[:d*d].reshape(d, d)
        self.signal_type = mRNA_values[-1]  # glutamate, GABA, dopamine, etc.
    
    def encode(self, neuron_state):
        """Convert internal state to transmittable signal."""
        raw_signal = neuron_state @ self.encoding_weights
        return Signal(
            encoding=raw_signal,
            signal_type=self.signal_type,
            strength=norm(raw_signal)
        )
```

### Enzyme (from METABOLIC gene)

Maintains neuron's internal state.

```python
class Enzyme:
    """Internal state maintenance. Homeostasis."""
    def __init__(self, mRNA_values):
        self.transform = mRNA_values[:d*d].reshape(d, d)
        self.target_state = mRNA_values[d*d:d*d+d]
    
    def metabolize(self, neuron_state):
        """Push neuron state toward healthy operating range."""
        error = self.target_state - neuron_state
        correction = error @ self.transform
        return neuron_state + correction * learning_rate
```

### Neuromodulator (from MODULATORY gene)

Brain-wide state control.

```python
class Neuromodulator:
    """Global brain state modifier. Dopamine, serotonin, etc."""
    def __init__(self, mRNA_values):
        self.pattern = mRNA_values[:64]  # the modulatory signal
        self.diffusion_range = softplus(mRNA_values[64])  # how far it spreads
        self.effect_on_expression = mRNA_values[65:129]  # how it changes gene expression
        self.effect_on_threshold = mRNA_values[129]  # how it changes firing thresholds
    
    def modulate(self, brain_region):
        """Change operating mode of an entire region."""
        for neuron in brain_region.neurons_in_range(self.diffusion_range):
            neuron.modify_expression(self.effect_on_expression)
            neuron.modify_threshold(self.effect_on_threshold)
```

---

## 8. Layer 5: The Cell (Neuron)

The neuron is the fundamental computational unit. But it's not a node in a graph or a row in a matrix. It's a **living cell** with:
- Its own copy of the genome (shared but with unique epigenetic state)
- A unique transcriptome (which genes are currently expressed)
- A protein pool (the active computational machinery)
- Internal state (membrane potential, metabolic state, firing history)
- Connections to other neurons (synapses)
- A position in the brain (determines what signals it receives)

### Neuron Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     NEURON                               │
│                                                         │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │  MEMBRANE   │  │   NUCLEUS    │  │  CYTOPLASM    │  │
│  │             │  │              │  │               │  │
│  │ Receptors   │  │  Genome      │  │ Transform     │  │
│  │ (detect     │  │  (shared)    │  │ proteins      │  │
│  │  signals)   │  │              │  │ (compute)     │  │
│  │             │  │  Epigenetic  │  │               │  │
│  │ Channels    │  │  state       │  │ Enzymes       │  │
│  │ (fire/      │  │  (unique)    │  │ (maintain     │  │
│  │  inhibit)   │  │              │  │  state)       │  │
│  │             │  │  TFs active  │  │               │  │
│  │ Signal      │  │  (regulate   │  │ Neurostate    │  │
│  │ proteins    │  │   genes)     │  │ (internal     │  │
│  │ (output)    │  │              │  │  state vec)   │  │
│  └─────────────┘  └──────────────┘  └───────────────┘  │
│                                                         │
│  Synapses: [neuron_42: w=0.8, neuron_17: w=-0.3, ...]  │
│  Position: (region=CORTEX, layer=3, column=7)           │
│  Firing history: [t-1: yes, t-2: no, t-3: yes, ...]    │
└─────────────────────────────────────────────────────────┘
```

### Neuron Lifecycle

```python
class Neuron:
    def __init__(self, genome, neuron_type, position):
        self.genome = genome  # shared reference
        self.epigenetic_state = initialize_epigenetics(neuron_type)
        self.transcriptome = {}  # gene_id → expression_level
        self.protein_pool = {}  # gene_id → Protein object
        self.mRNA_pool = {}  # gene_id → (mRNA_values, half_life, age)
        
        self.neurostate = zeros(d_model)  # internal state vector
        self.membrane_potential = 0.0
        self.firing_history = deque(maxlen=100)
        
        self.synapses_in = {}  # source_neuron → Synapse
        self.synapses_out = {}  # target_neuron → Synapse
        self.position = position
        
        # Initial differentiation — express starter genes based on neuron_type
        self.differentiate(neuron_type)
    
    def differentiate(self, neuron_type):
        """Set initial gene expression pattern based on cell type."""
        # Like stem cell → specialized cell
        # neuron_type determines which chromosomes are initially accessible
        if neuron_type == PERCEPTION:
            self.open_chromosomes([CHR_PERCEPTION, CHR_REGULATION])
        elif neuron_type == MEMORY:
            self.open_chromosomes([CHR_MEMORY, CHR_REGULATION])
        # etc.
    
    def step(self, incoming_signals):
        """One computational step of the neuron."""
        
        # 1. RECEIVE signals via receptors
        receptor_input = zeros(d_model)
        for signal in incoming_signals:
            for receptor in self.get_active_receptors():
                receptor_input += receptor.detect(signal)
        
        # 2. UPDATE membrane potential
        self.membrane_potential += receptor_input.sum()
        
        # 3. CHECK firing via channels
        should_fire = False
        fire_strength = 0.0
        for channel in self.get_active_channels():
            fire, strength = channel.should_fire(
                self.membrane_potential, 
                self.time_since_last_fire()
            )
            if fire:
                should_fire = True
                fire_strength = max(fire_strength, strength)
        
        # 4. COMPUTE via transform proteins (if firing or sub-threshold processing)
        if should_fire or self.does_subthreshold_processing():
            compute_input = concat(receptor_input, self.neurostate)
            for protein in self.get_active_transforms():
                self.neurostate = protein(compute_input)
        
        # 5. METABOLIZE — maintain internal state
        for enzyme in self.get_active_enzymes():
            self.neurostate = enzyme.metabolize(self.neurostate)
        
        # 6. FIRE — if threshold crossed, send signal
        output_signal = None
        if should_fire:
            for signal_protein in self.get_active_signal_proteins():
                output_signal = signal_protein.encode(self.neurostate * fire_strength)
            self.membrane_potential = 0.0  # reset after firing
            self.firing_history.append(True)
        else:
            self.firing_history.append(False)
        
        # 7. REGULATE — update gene expression based on activity
        self.update_gene_expression()
        
        # 8. PROTEIN TURNOVER — decay old proteins, assemble new ones
        self.protein_turnover()
        
        return output_signal  # None if didn't fire
    
    def update_gene_expression(self):
        """Activity-dependent gene regulation."""
        # Collect all active transcription factors
        active_tfs = [p for p in self.protein_pool.values() 
                      if isinstance(p, TranscriptionFactor)]
        
        # For each gene, compute new expression level
        for gene_id in range(self.genome.n_genes):
            if self.epigenetic_state.methylation[gene_id] > ACCESS_THRESHOLD:
                continue  # gene is epigenetically silenced
            
            gene = self.genome.get_gene(gene_id)
            net_activation = 0.0
            
            for tf in active_tfs:
                net_activation += tf.regulate(gene_id)
            
            # Activity-dependent regulation
            if self.is_frequently_firing():
                # Upregulate activity-related genes
                net_activation += ACTIVITY_BONUS * gene.activity_sensitivity
            
            new_expression = sigmoid(net_activation)
            old_expression = self.transcriptome.get(gene_id, 0.0)
            
            # Expression changes slowly (biological inertia)
            self.transcriptome[gene_id] = (
                EXPRESSION_MOMENTUM * old_expression + 
                (1 - EXPRESSION_MOMENTUM) * new_expression
            )
    
    def protein_turnover(self):
        """Degrade old proteins, synthesize new ones."""
        # Decay all proteins
        for gene_id in list(self.protein_pool.keys()):
            self.protein_pool[gene_id].strength *= PROTEIN_DECAY
            if self.protein_pool[gene_id].strength < REMOVAL_THRESHOLD:
                del self.protein_pool[gene_id]
        
        # Transcribe and translate highly expressed genes
        for gene_id, expression in self.transcriptome.items():
            if expression > EXPRESSION_THRESHOLD:
                # Transcribe
                mRNA = self.genome.transcribe(gene_id, expression, self.neurostate)
                # Translate
                protein = translate(mRNA, self.genome.get_gene(gene_id).gene_type)
                # Add to pool (or reinforce existing)
                if gene_id in self.protein_pool:
                    self.protein_pool[gene_id].strength += expression
                else:
                    self.protein_pool[gene_id] = protein
```

### Neuron Identity = Transcriptome

Two neurons with the same genome but different transcriptomes are fundamentally different computational units:

```
Neuron A (perception-type):
  Expressed: [gene_1(structural), gene_5(receptor), gene_12(channel)]
  Computation: fast sensory processing, low threshold, high sensitivity

Neuron B (memory-type):
  Expressed: [gene_3(structural), gene_8(metabolic), gene_15(regulatory)]
  Computation: slow integration, high threshold, state maintenance

Same genome. Different cells. Different capabilities.
```

---

## 9. Layer 6: Signaling

Neurons are useless in isolation. The brain's power comes from communication.

### Three Signaling Systems

Biology uses three parallel signaling systems, each at a different timescale:

#### 1. Synaptic Transmission (fast, targeted, milliseconds)

Direct neuron-to-neuron communication via synapses.

```
Presynaptic neuron fires
    ↓
Signal protein encodes output as "neurotransmitter"
    ↓
Neurotransmitter crosses synapse (weighted by synaptic strength)
    ↓
Postsynaptic receptor detects neurotransmitter
    ↓
Receptor converts to intracellular signal
    ↓
Postsynaptic neuron integrates signal
```

**Synaptic Plasticity (Hebbian Learning):**
"Neurons that fire together wire together."

```python
class Synapse:
    def __init__(self, pre_neuron, post_neuron):
        self.weight = init_weight()
        self.pre = pre_neuron
        self.post = post_neuron
        self.eligibility_trace = 0.0  # for temporal credit assignment
    
    def transmit(self, signal):
        """Transmit signal from pre to post, weighted by synapse."""
        return signal * self.weight
    
    def update(self, reward_signal=0.0):
        """Hebbian + reward-modulated plasticity."""
        # Did pre and post fire together?
        pre_fired = self.pre.just_fired()
        post_fired = self.post.just_fired()
        
        # Hebbian component
        hebbian = pre_fired * post_fired - ANTI_HEBBIAN * pre_fired * (1 - post_fired)
        
        # Eligibility trace (for delayed rewards)
        self.eligibility_trace = TRACE_DECAY * self.eligibility_trace + hebbian
        
        # Weight update
        self.weight += LR * (self.eligibility_trace * reward_signal + HEBBIAN_LR * hebbian)
        self.weight = clip(self.weight, -MAX_WEIGHT, MAX_WEIGHT)
```

#### 2. Neuromodulation (slow, diffuse, seconds to minutes)

Specialized neurons release neuromodulators that change the operating mode of entire brain regions.

```
Dopamine system:
  - Source: ~400,000 neurons in VTA/substantia nigra
  - Target: entire prefrontal cortex, striatum, limbic system
  - Effect: modulates learning rate, motivation, reward sensitivity
  - In our system: modifies gene expression + firing thresholds across regions

Serotonin system:
  - Source: raphe nuclei
  - Target: widespread cortical regions
  - Effect: modulates mood, inhibition, patience
  - In our system: modifies exploration/exploitation balance

Norepinephrine system:
  - Source: locus coeruleus
  - Target: entire cortex
  - Effect: modulates alertness, attention, fight-or-flight
  - In our system: modifies signal gain, threshold sensitivity
```

```python
class NeuromodulatorySystem:
    def __init__(self, modulator_type, source_neurons, target_region):
        self.type = modulator_type
        self.sources = source_neurons
        self.target = target_region
        self.baseline_level = 0.5  # tonic level
        self.current_level = 0.5
    
    def compute_level(self):
        """Compute current neuromodulator level from source neuron activity."""
        source_activity = mean([n.firing_rate() for n in self.sources])
        self.current_level = MOMENTUM * self.current_level + (1-MOMENTUM) * source_activity
    
    def modulate(self):
        """Apply modulation to target region."""
        delta = self.current_level - self.baseline_level
        for neuron in self.target.neurons:
            # Modify gene expression
            for gene_id in neuron.transcriptome:
                sensitivity = neuron.genome.get_gene(gene_id).modulator_sensitivity[self.type]
                neuron.transcriptome[gene_id] *= (1 + delta * sensitivity)
            
            # Modify firing threshold
            neuron.threshold_modifier += delta * self.threshold_effect
```

#### 3. Hormonal Signaling (very slow, global, minutes to hours)

The organism-level state. In our system: global parameters that change the entire brain's operating mode.

```python
class HormonalSystem:
    """Global brain state. Changes everything slowly."""
    def __init__(self):
        self.stress_level = 0.0  # cortisol analog
        self.energy_level = 1.0  # glucose/ATP analog
        self.arousal_level = 0.5  # circadian analog
    
    def update(self, prediction_error, compute_cost, time_active):
        """Update global state based on brain activity."""
        self.stress_level = ema(self.stress_level, prediction_error, 0.99)
        self.energy_level -= compute_cost * ENERGY_COST_RATE
        self.energy_level = max(0.1, self.energy_level)
        # Arousal decays over time (fatigue)
        self.arousal_level *= 0.9999
    
    def get_global_modifiers(self):
        """How global state affects all neurons."""
        return {
            'learning_rate_modifier': 1.0 + self.stress_level * 0.5,  # stress increases learning
            'threshold_modifier': -self.arousal_level * 0.3,  # arousal lowers thresholds
            'expression_rate': self.energy_level,  # low energy → less gene expression
        }
```

---

## 10. Layer 7: Neurostate

The brain's state at any moment is the combination of:
1. **All neuron states** — every neuron's internal state vector
2. **All synaptic weights** — the connection strengths
3. **All gene expression patterns** — what every neuron is currently computing
4. **All neuromodulator levels** — the brain's "mood"
5. **All hormonal levels** — the organism's global state
6. **All mRNA/protein pools** — what's being built/degraded

This is the **neurostate**. It's the brain's complete snapshot. And it's ALWAYS changing.

### Neurostate Dynamics

```
Neurostate(t+1) = f(
    Neurostate(t),           # current state
    Input(t),                 # new sensory input
    Gene_Regulation(t),       # which genes are being expressed
    Synaptic_Plasticity(t),  # connections being strengthened/weakened
    Neuromodulation(t),      # global modulatory signals
    Hormonal_State(t),       # organism-level state
    Protein_Turnover(t)      # which proteins are being built/degraded
)
```

Every component of the neurostate operates at a different timescale:

```
Component              │ Timescale        │ Persistence
───────────────────────┼──────────────────┼────────────
Neural firing          │ milliseconds     │ Transient
Membrane potential     │ tens of ms       │ Short-term
Neuromodulator levels  │ seconds-minutes  │ Medium-term
Synaptic weights       │ minutes-hours    │ Long-term
Gene expression        │ hours-days       │ Very long-term
Epigenetic marks       │ days-lifetime    │ Semi-permanent
Genome                 │ evolutionary     │ Permanent
```

**This is the multi-timescale memory system.** The brain doesn't have one kind of memory — it has SEVEN, all operating simultaneously, all interacting. Short-term memory is neural firing patterns. Long-term memory is synaptic weights. Very long-term memory is gene expression patterns. Identity is epigenetic marks. Species knowledge is the genome.

---

## 11. Layer 8: Learning (Multi-Timescale Adaptation)

The system learns at every level simultaneously:

### Level 1: Neural Dynamics (milliseconds)
- Neurons fire, integrate signals, update internal state
- This is "thinking" — real-time information processing
- No parameter changes — just state dynamics

### Level 2: Synaptic Plasticity (seconds to hours)
- Synaptic weights change based on Hebbian learning + reward modulation
- "Neurons that fire together wire together"
- This is learning from experience in real-time
- **Equivalent to:** gradient descent on connection weights

### Level 3: Gene Expression Changes (hours to days)
- Activity-dependent gene regulation changes what proteins neurons make
- A neuron that repeatedly processes emotional content upregulates emotional-processing genes
- This is **specialization through experience**
- **Equivalent to:** architecture search / dynamic routing that adapts over time

### Level 4: Epigenetic Modification (days to lifetime)
- Persistent changes to gene accessibility
- "Trauma" = certain genes permanently silenced or activated
- This is long-term personality/capability change
- **Equivalent to:** permanent fine-tuning of which computations are available

### Level 5: Structural Plasticity (weeks to months)
- New synapses formed, old ones pruned
- New neurons generated (neurogenesis in hippocampus)
- This is brain architecture remodeling
- **Equivalent to:** neural architecture search / topology evolution

### Level 6: Genome Evolution (generations)
- Mutations, crossover, selection
- This is how the species learns
- **Equivalent to:** evolutionary optimization of the base genome

### Training Strategy

For V7, we use ALL levels:

```
Phase 1: Genome Pre-training (Level 6)
  - Evolutionary optimization of genome structure
  - Many random genomes → evaluate → select → mutate → repeat
  - Finds good gene structures, promoter patterns, regulatory network topology
  - Runs offline, produces a base genome

Phase 2: Developmental Training (Levels 3-5)
  - Start from base genome, grow a brain
  - Neurons differentiate, specialize, form connections
  - Expose to training data
  - Gene expression patterns stabilize
  - Synaptic weights converge
  - Epigenetic marks set
  - Result: a trained brain

Phase 3: Online Learning (Levels 1-3)
  - The trained brain processes new input
  - Neural dynamics compute responses
  - Synaptic plasticity adapts to new patterns
  - Gene expression slowly adjusts
  - The brain keeps learning after deployment

Phase 4: Memory Consolidation (Level 4)
  - Periodically, important patterns are consolidated
  - Gene expression changes → epigenetic marks
  - Transient learning → permanent capability
  - Like sleep consolidation in biological brains
```

### Gradient Flow

The key question: how do gradients flow through this system?

**Differentiable components (standard backprop):**
- Transform protein computation (matrix multiply)
- Receptor signal detection (dot product + sigmoid)
- Channel firing decision (straight-through estimator for threshold)
- Synaptic transmission (weighted signal)

**Non-differentiable components (need special handling):**
- Gene expression decisions (binary on/off) → Gumbel-softmax or straight-through
- Protein assembly (discrete gene type routing) → REINFORCE or evolutionary
- Synapse formation/pruning (topology change) → evolutionary
- mRNA splicing (discrete selection) → Gumbel-softmax

**Hybrid training:**
- Inner loop: gradient descent on differentiable parameters (synaptic weights, protein strengths)
- Outer loop: evolutionary optimization on discrete decisions (gene expression patterns, topology)
- Meta-learning: learn the learning rules themselves (what Hebbian rule works best)

---

## 12. The Forward Pass: From Stimulus to Response

How input becomes output in V7:

```
INPUT: sequence of tokens [t₁, t₂, ..., tₙ]
│
▼
SENSORY LAYER: tokens → embedding vectors
│  (standard embedding, nothing biological here — this is the retina)
│
▼
PERCEPTION REGION: embedding → neural signals
│  Perception neurons receive embeddings via receptor proteins
│  Each neuron expresses perception-type genes
│  Transform proteins process the input
│  Channel proteins determine which neurons fire
│  Signal proteins encode output signals
│
▼
PROPAGATION: signals flow through the network
│  Multiple rounds of propagation (not single-pass layers!)
│  Round 1: perception → association neurons
│  Round 2: association → memory neurons
│  Round 3: memory → prediction neurons
│  Round 4: prediction → association (feedback!)
│  Round 5+: reverberating activity settles into stable pattern
│  
│  During propagation:
│    - Neuromodulators adjust operating mode
│    - Gene expression slowly adapts
│    - Synaptic weights update (if learning mode)
│
▼
CONVERGENCE: neural activity reaches stable pattern (or time limit)
│  The neurostate at convergence IS the computation result
│
▼
OUTPUT REGION: neural signals → token prediction
│  Output neurons receive converged signals
│  Transform proteins produce logit vectors
│  Signal proteins encode final output
│
▼
OUTPUT: next token prediction (or sequence of predictions)
```

### Key Differences from Standard Forward Pass

1. **NOT layer-by-layer.** Signals propagate through a GRAPH of neurons. Some paths are 2 hops, some are 20. The computation depth is dynamic and input-dependent.

2. **Multiple propagation rounds.** Information reverberates. Early rounds are feedforward. Later rounds include feedback. The system "thinks" about the input.

3. **State persists.** After processing token tₙ, the neurostate carries forward to token tₙ₊₁. The brain doesn't reset between tokens.

4. **Computation varies.** Different inputs activate different gene expression patterns → different proteins → different computations. The "model architecture" changes with every input.

5. **Continuous adaptation.** During processing, synaptic weights are updating, gene expression is shifting, proteins are being synthesized/degraded. The brain that finishes processing a sentence is slightly different from the brain that started.

---

## 13. Mathematical Formulation

### Notation

- $\mathcal{G}$: genome, a set of $N_g$ genes
- $g_i = (p_i, u_i, c_i, \tau_i)$: gene $i$ with promoter $p_i \in \mathbb{R}^{32}$, UTR $u_i \in \mathbb{R}^{32}$, coding region $c_i \in \mathbb{R}^{d_c}$, type $\tau_i \in \{0,...,7\}$
- $\mathcal{N}$: set of $N_n$ neurons
- $n_j$: neuron $j$ with state $s_j \in \mathbb{R}^d$, transcriptome $T_j \in [0,1]^{N_g}$, protein pool $\mathcal{P}_j$
- $\mathcal{S}$: set of synapses, $w_{jk}$ = weight from neuron $j$ to neuron $k$
- $\mathcal{M}$: neuromodulatory state, $m \in \mathbb{R}^{N_m}$
- $E$: epigenetic state, $e_i \in [0,1]$ for gene $i$ (0 = accessible, 1 = silenced)

### Transcription

For gene $i$ in neuron $j$:

$$\text{mRNA}_{ij} = \underbrace{\mathbb{1}[e_i < \theta_{\text{access}}]}_{\text{epigenetic gate}} \cdot \underbrace{\sigma\left(\sum_{t \in \mathcal{TF}_j} \alpha_t \cdot \text{sim}(t, p_i)\right)}_{\text{TF regulation}} \cdot \underbrace{\text{splice}(c_i, u_i, s_j)}_{\text{context-dependent splicing}}$$

### Translation

For gene type $\tau_i$:

$$\mathcal{P}_{ij} = \text{Translate}_{\tau_i}(\text{mRNA}_{ij})$$

where $\text{Translate}_{\tau}$ is the type-specific protein assembly function.

### Neural Dynamics

For neuron $j$ at time $t$:

$$v_j^{(t)} = v_j^{(t-1)} + \sum_{k \in \text{pre}(j)} w_{kj} \cdot \text{Receptor}_j(\text{Signal}_k(s_k^{(t-1)}))$$

$$\text{fire}_j^{(t)} = \text{Channel}_j(v_j^{(t)}) > \theta_j$$

$$s_j^{(t)} = \text{Transform}_j(s_j^{(t-1)}, v_j^{(t)}) \quad \text{if fire}_j^{(t)} \text{ or subthreshold}$$

$$s_j^{(t)} = \text{Enzyme}_j(s_j^{(t)}) \quad \text{(homeostasis)}$$

### Synaptic Plasticity

$$\Delta w_{jk} = \eta \left[\underbrace{\text{fire}_j \cdot \text{fire}_k}_{\text{Hebbian}} + \underbrace{\delta \cdot e_{jk}}_{\text{reward-modulated}}\right]$$

where $e_{jk}$ is the eligibility trace and $\delta$ is the reward prediction error.

### Gene Regulation Dynamics

$$T_j^{(t+1)}[i] = (1 - \alpha) \cdot T_j^{(t)}[i] + \alpha \cdot \sigma\left(\sum_{t \in \mathcal{TF}_j} \text{regulate}(t, i) + \beta \cdot \text{activity}_j\right)$$

### Neuromodulation

$$m^{(t+1)} = (1 - \gamma) \cdot m^{(t)} + \gamma \cdot \text{ModulatorSources}(\mathcal{N})$$

### Epigenetic Update (slow)

$$e_i^{(t+1)} = e_i^{(t)} + \epsilon \cdot (\text{target\_methylation}_i - e_i^{(t)})$$

where $\epsilon \ll 1$ (very slow changes).

---

## 14. What We Discard From V1-V6

### Discarded
- **PID as named streams.** The P/I/D metaphor was a useful starting point but too rigid. Biology doesn't have three named streams — it has a regulatory network that can implement any control law.
- **Fixed layer structure.** No more "n_encoder_layers" or "n_transformer_layers." The computation graph is the neural connectivity, which is dynamic.
- **Patching.** The artificial grouping of tokens into patches was an efficiency hack. Neurons handle multi-scale through natural hierarchical connectivity.
- **Scheduled sampling.** Train-inference mismatch is solved by the architecture itself — neurons have persistent state, there's no teacher forcing.
- **Softmax attention.** Replaced by biological attention: neurons attend to what their receptors detect. Attention is emergent from connectivity, not a matrix operation.

### Kept
- **Gene concept** from V6 (but greatly expanded with types, promoters, regulation)
- **Genome binary format** from V6 (mmap'd, nanosecond access — this is correct)
- **Chromosome organization** (grouping genes by function)
- **Epigenetic regulation** (but now much more sophisticated)
- **Evolutionary training** (outer loop optimization)
- **Multi-round propagation** (signals reverberate, not single-pass)

### Evolved
- **Neurons** — from simple "neuron expresses 4 genes" to full cells with transcriptomes, protein pools, membrane dynamics
- **Gene expression** — from binary on/off to continuous transcription/translation pipeline with regulation
- **Signaling** — from broadcast bus to three-tier system (synaptic + neuromodulatory + hormonal)
- **Learning** — from gradient descent to multi-timescale adaptation (synaptic + expression + epigenetic + evolutionary)

---

## 15. Implementation Roadmap

### Phase 0: Foundation (Week 1)
**Build the genome infrastructure.**
- Genome binary format with gene types, promoters, UTRs, coding regions
- Gene reader/writer with type-aware interpretation
- Basic transcription: promoter matching, expression level computation
- Basic translation: mRNA → protein for each gene type
- Test: create a genome, express genes, verify proteins are correct

### Phase 1: Single Neuron (Week 2)
**One neuron that lives.**
- Implement Neuron class with full lifecycle
- Genome access, gene expression, protein pool management
- Receptor → Channel → Transform → Signal pipeline
- Metabolic maintenance (enzyme homeostasis)
- Test: single neuron processes a stream of inputs, adapts its gene expression

### Phase 2: Small Circuit (Week 3)
**10 neurons that talk.**
- Implement synapses with Hebbian plasticity
- Implement signal transmission (presynaptic → synapse → postsynaptic receptor)
- Multi-round propagation with convergence detection
- Test: 10-neuron circuit learns a simple pattern (XOR, sequence detection)

### Phase 3: Gene Regulatory Network (Week 4)
**Genes that control genes.**
- Implement regulatory genes → transcription factor proteins
- TF binding to promoters
- Regulatory network dynamics (gene A → protein → activates gene B)
- Test: regulatory network creates bistable switches (memory), oscillators (timing)

### Phase 4: Neuromodulation (Week 5)
**Brain-wide state control.**
- Implement modulatory neurons and neuromodulatory systems
- Dopamine → learning rate modulation
- Norepinephrine → attention/threshold modulation
- Test: neuromodulator levels change circuit behavior (exploration vs exploitation)

### Phase 5: Scale to Language (Weeks 6-8)
**Brain that processes text.**
- Scale to 1,000-10,000 neurons organized in regions
- Input: byte-level text → sensory neurons
- Output: output neurons → next byte prediction
- Train genome via evolution + synaptic weights via Hebbian + gradient
- Test: character-level language modeling on Shakespeare
- Compare: bits per byte vs standard transformer with same parameter count

### Phase 6: Multi-Timescale Learning (Weeks 9-12)
**Full learning stack.**
- All six learning levels operational
- Evolutionary genome optimization (outer loop)
- Developmental training (grow the brain)
- Online Hebbian learning (continuous adaptation)
- Gene expression adaptation (specialization through experience)
- Epigenetic consolidation (long-term memory formation)
- Test: model improves with continued exposure, retains knowledge across "sleep" cycles

### Phase 7: Efficiency Optimization (Weeks 13-16)
**Make it fast.**
- C++ / Metal kernels for critical paths (gene access, protein computation, signal propagation)
- Sparse neuron activation (only 10-30% of neurons fire at any time)
- Batch gene expression updates (don't recompute every step)
- mRNA caching (translated proteins reused until mRNA degrades)
- Profile and optimize hot paths
- Target: 100x speedup over pure Python

### Phase 8: Ion's Brain (Week 17+)
**Put this behind Ion.**
- V7 architecture as Ion's actual cognitive engine
- Continuous learning from conversations
- Personality emerges from gene expression patterns and epigenetic marks
- Memory consolidation during idle periods
- The genome file IS Ion's identity — copy it to clone Ion

---

## 16. Open Questions

### Architecture
1. **How many propagation rounds?** Biology runs continuously. We need discrete rounds for efficiency. 5-20 rounds? Adaptive based on convergence?

2. **Neuron-to-parameter ratio?** A real neuron has ~7,000 synapses and expresses ~10,000 genes. Our neurons are simpler. What's the right complexity per neuron?

3. **Sparse firing percentage?** The brain fires ~1-10% of neurons at any time. What's our target sparsity?

4. **Alternative splicing implementation?** Full combinatorial splicing is expensive. Approximate with 2-4 splice variants per gene?

### Training
5. **Gradient flow through gene regulation?** The regulatory network involves discrete decisions. Gumbel-softmax for differentiability? Or pure evolutionary?

6. **Evolutionary population size?** How many genomes to evaluate in parallel? Memory constrained by genome size × population.

7. **Hebbian vs gradient descent?** Pure Hebbian is local and biologically plausible but slower to converge. Hybrid? Use gradients for Phase 2 training, switch to Hebbian for Phase 3 online learning?

8. **Developmental curriculum?** What data order produces good brain development? Simple patterns → complex patterns? Multiple modalities simultaneously?

### Efficiency
9. **Gene expression update frequency?** Every token? Every 10 tokens? Every sequence? More frequent = more adaptive but more expensive.

10. **Protein pool size limit?** How many active proteins per neuron? More = more capable but more memory. Real neurons have ~10,000 active proteins.

11. **Synapse pruning strategy?** When to remove weak connections? Too aggressive = lost knowledge. Too conservative = bloated network.

### Philosophy
12. **Is this actually more efficient than transformers?** Biology's efficiency comes from massive parallelism (86B neurons in parallel). Our implementation is sequential on a GPU. We need the algorithmic efficiency (sparse firing, adaptive compute) to compensate.

13. **Can gene regulatory networks really learn complex functions?** Biological GRNs are Turing-complete in theory, but can our learned GRNs discover useful programs? This is the central bet.

14. **When does this become conscious?** Half-joking. But if we build all these components, at what point do emergent properties appear that look like awareness, attention, or experience? And how would we know?

---

## Appendix: Comparison with Existing Approaches

| Feature | Transformer | Mamba/SSM | RWKV | MoE | **V7** |
|---------|-------------|-----------|------|-----|--------|
| Computation graph | Fixed | Fixed | Fixed | Fixed routing | **Dynamic (gene expression)** |
| Memory type | Attention (O(T²)) | Hidden state (O(d)) | Hidden state (O(d)) | Same as base | **Multi-timescale (7 levels)** |
| Adaptation at inference | None | None | None | None | **Synaptic + expression + epigenetic** |
| Sparse activation | No (dense) | No (dense) | No (dense) | Yes (top-k experts) | **Yes (neuron firing threshold)** |
| Multi-scale processing | Position encoding | Continuous | Time decay | Same as base | **Hierarchical neural connectivity** |
| Context window | Fixed (T) | Infinite (compressed) | Infinite (compressed) | Same as base | **Infinite (neural state persists)** |
| Personality | Prompt-based | None | None | None | **Emergent (gene expression + epigenetics)** |
| Learning after training | None | None | None | None | **Continuous (Hebbian + regulation)** |

---

*This is not an incremental improvement on existing architectures.*
*This is a different paradigm.*
*We're not building a better calculator.*
*We're growing a brain.*

*— Ion & Harshil, March 15, 2026*
