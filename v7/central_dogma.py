"""
V7 Central Dogma: DNA → RNA → Protein → Function

The transcription/translation pipeline that V1-V6 completely skipped.
This is where regulation, context-dependence, and adaptive computation live.

Pipeline:
1. Accessibility check (epigenetic gating)
2. Transcription factor binding (regulatory network)
3. Transcription (gene → mRNA with expression level)
4. Alternative splicing (context-dependent mRNA variants)
5. Translation (mRNA → protein)
6. Post-translational modification (activity-dependent scaling)
7. Protein deployment (to neuron compartments)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from genome import Genome, GeneType, GeneEntry


# ============================================================
# Proteins — The Functional Units
# ============================================================

@dataclass
class Protein:
    """Base class for all proteins."""
    gene_id: int
    gene_type: GeneType
    strength: float = 1.0        # post-translational modification (decays over time)
    age: int = 0                  # steps since creation
    
    def decay(self, rate: float = 0.995):
        """Proteins degrade over time."""
        self.strength *= rate
        self.age += 1
    
    @property
    def is_alive(self) -> bool:
        return self.strength > 0.01


@dataclass
class TransformProtein(Protein):
    """
    From STRUCTURAL gene. The workhorse.
    Encodes a linear transform: y = Wx + b
    """
    W: Optional[np.ndarray] = None   # (d_in, d_out) float32
    b: Optional[np.ndarray] = None   # (d_out,) float32
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply transform."""
        out = x @ self.W
        if self.b is not None:
            out = out + self.b
        return out * self.strength


@dataclass
class TranscriptionFactor(Protein):
    """
    From REGULATORY gene. Controls other genes.
    Binds to gene promoters and activates/represses transcription.
    """
    binding_pattern: Optional[np.ndarray] = None   # (128,) float32 — what promoters to match
    target_genes: Optional[np.ndarray] = None       # (n_targets,) int — which genes to affect
    regulation_strength: Optional[np.ndarray] = None # (n_targets,) float32 — +activate, -repress
    is_activator: bool = True
    
    def bind_affinity(self, promoter: np.ndarray) -> float:
        """
        Compute binding affinity to a gene's promoter.
        Uses cosine similarity — higher = better match.
        """
        if self.binding_pattern is None:
            return 0.0
        # Truncate or pad to match
        bp = self.binding_pattern
        pr = promoter.astype(np.float32)
        min_len = min(len(bp), len(pr))
        bp = bp[:min_len]
        pr = pr[:min_len]
        
        norm_bp = np.linalg.norm(bp) + 1e-8
        norm_pr = np.linalg.norm(pr) + 1e-8
        cos_sim = np.dot(bp, pr) / (norm_bp * norm_pr)
        return float(cos_sim) * self.strength


@dataclass
class ReceptorProtein(Protein):
    """
    From RECEPTOR gene. Sits on neuron membrane.
    Detects specific incoming signals (neurotransmitters).
    """
    selectivity: Optional[np.ndarray] = None    # (d,) float32 — what signals to respond to
    sensitivity: float = 0.5                      # how responsive
    excitatory: bool = True                       # excitatory or inhibitory
    
    def detect(self, signal: np.ndarray) -> float:
        """Compute response to incoming signal."""
        if self.selectivity is None:
            return 0.0
        sel = self.selectivity
        sig = signal.astype(np.float32)
        min_len = min(len(sel), len(sig))
        
        match = np.dot(sel[:min_len], sig[:min_len])
        match /= (np.linalg.norm(sel[:min_len]) + 1e-8) * (np.linalg.norm(sig[:min_len]) + 1e-8)
        
        response = max(0.0, match) * self.sensitivity * self.strength
        return response if self.excitatory else -response


@dataclass 
class ChannelProtein(Protein):
    """
    From CHANNEL gene. Controls neuron firing.
    Implements threshold-based gating with refractory period.
    """
    threshold: float = 0.5
    conductance: float = 1.0
    refractory_period: float = 1.0    # steps before can fire again
    gate_weights: Optional[np.ndarray] = None  # (d,) for voltage-dependent gating
    
    def should_fire(self, membrane_potential: float, 
                    time_since_fire: float) -> Tuple[bool, float]:
        """Determine if neuron should fire."""
        if time_since_fire < self.refractory_period:
            return False, 0.0
        
        effective_threshold = self.threshold / (self.strength + 1e-8)
        if membrane_potential > effective_threshold:
            output_strength = min(membrane_potential, 5.0) * self.conductance * self.strength
            return True, output_strength
        return False, 0.0


@dataclass
class SignalProtein(Protein):
    """
    From SIGNAL gene. Neurotransmitter synthesis.
    Encodes neuron output as transmittable signal.
    """
    encoding_W: Optional[np.ndarray] = None   # (d_in, d_out) encoding matrix
    signal_type: int = 0                        # neurotransmitter type
    
    def encode(self, state: np.ndarray) -> np.ndarray:
        """Convert internal state to output signal."""
        if self.encoding_W is None:
            return state * self.strength
        return (state @ self.encoding_W) * self.strength


@dataclass
class Enzyme(Protein):
    """
    From METABOLIC gene. Internal state maintenance.
    Pushes neuron state toward healthy operating range.
    """
    transform: Optional[np.ndarray] = None     # (d, d) state transform
    target_state: Optional[np.ndarray] = None  # (d,) homeostatic target
    correction_rate: float = 0.01
    
    def metabolize(self, state: np.ndarray) -> np.ndarray:
        """Apply homeostatic correction."""
        if self.target_state is None:
            return state
        
        error = self.target_state - state
        if self.transform is not None:
            correction = error @ self.transform
        else:
            correction = error
        
        return state + correction * self.correction_rate * self.strength


@dataclass
class Neuromodulator(Protein):
    """
    From MODULATORY gene. Brain-wide state control.
    Released by specialized neurons, affects entire regions.
    """
    pattern: Optional[np.ndarray] = None        # (64,) modulatory signal
    diffusion_range: float = 1.0                  # how far it spreads (0-1 = fraction of brain)
    expression_effect: Optional[np.ndarray] = None  # how it changes gene expression
    threshold_effect: float = 0.0                 # how it changes firing thresholds


# ============================================================
# mRNA — The Intermediary
# ============================================================

@dataclass
class mRNA:
    """
    Messenger RNA — the working copy of a gene.
    Has a finite lifetime (half-life). Decays if not reinforced.
    Can be alternatively spliced based on context.
    """
    gene_id: int
    gene_type: GeneType
    values: np.ndarray          # the transcribed values (float32)
    expression_level: float     # how strongly expressed (0-1)
    halflife: float             # decay rate per step
    age: int = 0
    
    def decay(self) -> bool:
        """Decay mRNA. Returns True if still alive."""
        self.expression_level *= self.halflife
        self.age += 1
        return self.expression_level > 0.01
    
    def reinforce(self, new_expression: float):
        """Reinforce mRNA with new transcription."""
        self.expression_level = max(self.expression_level, new_expression)
        self.age = 0  # reset age on reinforcement


# ============================================================
# Transcription Engine
# ============================================================

class TranscriptionEngine:
    """
    Implements the Central Dogma: DNA → mRNA → Protein
    
    This is the core regulatory machinery that was completely missing in V1-V6.
    """
    
    # Thresholds
    ACCESS_THRESHOLD = 0.5         # methylation above this = gene silenced
    BINDING_THRESHOLD = 0.1        # TF affinity below this = no binding
    EXPRESSION_THRESHOLD = 0.05    # expression below this = not transcribed
    
    def __init__(self, genome: Genome):
        self.genome = genome
    
    # ---- STEP 1: Accessibility ----
    
    def is_accessible(self, gene_id: int) -> bool:
        """Check if gene is epigenetically accessible."""
        return self.genome.get_methylation(gene_id) < self.ACCESS_THRESHOLD
    
    # ---- STEP 2: TF Binding ----
    
    def compute_tf_activation(self, gene_id: int, 
                               active_tfs: List[TranscriptionFactor]) -> float:
        """
        Compute net activation of a gene from all active transcription factors.
        Positive = activated, Negative = repressed.
        """
        if not active_tfs:
            # No TFs → use base expression
            return self.genome.get_gene(gene_id).base_expression
        
        promoter = self.genome.get_promoter(gene_id).astype(np.float32)
        
        net_activation = 0.0
        for tf in active_tfs:
            affinity = tf.bind_affinity(promoter)
            if abs(affinity) > self.BINDING_THRESHOLD:
                if tf.is_activator:
                    net_activation += affinity
                else:
                    net_activation -= affinity
        
        # Add base expression as prior
        base = self.genome.get_gene(gene_id).base_expression
        return base + net_activation
    
    # ---- STEP 3: Transcription ----
    
    def transcribe(self, gene_id: int, expression_level: float,
                   context: Optional[np.ndarray] = None) -> Optional[mRNA]:
        """
        Transcribe a gene into mRNA.
        
        Args:
            gene_id: which gene to transcribe
            expression_level: how strongly to express (0-1)
            context: current neuron state (for context-dependent splicing)
        
        Returns:
            mRNA object, or None if expression too low
        """
        if expression_level < self.EXPRESSION_THRESHOLD:
            return None
        
        gene = self.genome.get_gene(gene_id)
        
        # Read coding region
        coding = self.genome.get_coding_region(gene_id).astype(np.float32)
        
        # Apply expression level modulation
        # Higher expression → more faithful transcription
        # Lower expression → noisier transcription (biological noise)
        noise_level = (1.0 - expression_level) * 0.05
        if noise_level > 0:
            coding = coding + np.random.randn(*coding.shape).astype(np.float32) * noise_level
        
        # Alternative splicing (Step 4)
        if gene.n_splice_variants > 1 and context is not None:
            coding = self._splice(coding, gene, context)
        
        return mRNA(
            gene_id=gene_id,
            gene_type=GeneType(gene.gene_type),
            values=coding,
            expression_level=expression_level,
            halflife=gene.mRNA_halflife,
        )
    
    # ---- STEP 4: Alternative Splicing ----
    
    def _splice(self, coding: np.ndarray, gene: GeneEntry, 
                context: np.ndarray) -> np.ndarray:
        """
        Alternative splicing: same gene, different mRNA based on context.
        
        Uses the gene's UTR to determine splice pattern.
        Different contexts → different parts of the coding region retained.
        """
        utr = self.genome.get_utr(gene.id).astype(np.float32)
        
        # UTR encodes splice decision boundaries
        # Context projects onto UTR to determine which "exons" to include
        ctx_truncated = context[:min(len(context), len(utr))]
        utr_truncated = utr[:len(ctx_truncated)]
        
        # Splice score per codon-sized block
        block_size = 16  # codon size
        n_blocks = len(coding) // block_size
        if n_blocks <= 1:
            return coding
        
        # Compute inclusion probability for each block
        # Using a simple hash of context × UTR
        splice_signal = np.dot(ctx_truncated, utr_truncated)
        splice_probs = np.ones(n_blocks, dtype=np.float32)
        
        # Some blocks are always included (exons), some are context-dependent (alternative exons)
        for i in range(n_blocks):
            phase = float(i) / n_blocks * np.pi * 2
            splice_probs[i] = 0.5 + 0.5 * np.tanh(splice_signal * np.sin(phase + splice_signal))
        
        # Apply splice mask (soft masking — scale blocks by inclusion probability)
        spliced = coding.copy()
        for i in range(n_blocks):
            start = i * block_size
            end = min(start + block_size, len(spliced))
            spliced[start:end] *= splice_probs[i]
        
        return spliced
    
    # ---- STEP 5: Translation ----
    
    def translate(self, mrna: mRNA) -> Optional[Protein]:
        """
        Translate mRNA into a protein.
        
        Different gene types → different protein types.
        """
        if mrna is None:
            return None
        
        gene = self.genome.get_gene(mrna.gene_id)
        values = mrna.values
        
        if mrna.gene_type == GeneType.STRUCTURAL:
            return self._translate_structural(mrna, gene)
        elif mrna.gene_type == GeneType.REGULATORY:
            return self._translate_regulatory(mrna, gene)
        elif mrna.gene_type == GeneType.RECEPTOR:
            return self._translate_receptor(mrna, gene)
        elif mrna.gene_type == GeneType.CHANNEL:
            return self._translate_channel(mrna, gene)
        elif mrna.gene_type == GeneType.SIGNAL:
            return self._translate_signal(mrna, gene)
        elif mrna.gene_type == GeneType.METABOLIC:
            return self._translate_metabolic(mrna, gene)
        elif mrna.gene_type == GeneType.MODULATORY:
            return self._translate_modulatory(mrna, gene)
        
        return None
    
    def _translate_structural(self, mrna: mRNA, gene: GeneEntry) -> TransformProtein:
        """mRNA → weight matrix + bias."""
        d_in, d_out = gene.d_in, gene.d_out
        vals = mrna.values
        
        W_size = d_in * d_out
        if len(vals) >= W_size + d_out:
            W = vals[:W_size].reshape(d_in, d_out)
            b = vals[W_size:W_size + d_out]
        elif len(vals) >= W_size:
            W = vals[:W_size].reshape(d_in, d_out)
            b = np.zeros(d_out, dtype=np.float32)
        else:
            # Fallback: use what we have, pad with zeros
            W = np.zeros((d_in, d_out), dtype=np.float32)
            W.flat[:len(vals)] = vals[:min(len(vals), d_in * d_out)]
            b = np.zeros(d_out, dtype=np.float32)
        
        return TransformProtein(
            gene_id=mrna.gene_id,
            gene_type=GeneType.STRUCTURAL,
            strength=mrna.expression_level,
            W=W,
            b=b,
        )
    
    def _translate_regulatory(self, mrna: mRNA, gene: GeneEntry) -> TranscriptionFactor:
        """mRNA → transcription factor."""
        vals = mrna.values
        
        # Binding pattern: first 128 values
        bp_size = min(128, len(vals))
        binding_pattern = np.zeros(128, dtype=np.float32)
        binding_pattern[:bp_size] = vals[:bp_size]
        
        # Target genes: next 96 values → interpret as gene IDs (top-k by magnitude)
        if len(vals) > 128:
            target_vals = vals[128:min(224, len(vals))]
            # Interpret as preferences — top values → target gene IDs
            n_targets = min(8, self.genome.n_genes)
            # Use absolute values, map to gene indices
            target_scores = np.abs(target_vals[:min(len(target_vals), self.genome.n_genes)])
            target_indices = np.argsort(target_scores)[-n_targets:]
            target_genes = target_indices.astype(np.int32)
        else:
            target_genes = np.array([], dtype=np.int32)
        
        # Strength: next 64 values
        if len(vals) > 224:
            strength_vals = vals[224:min(288, len(vals))]
            reg_strength = strength_vals[:len(target_genes)] if len(strength_vals) >= len(target_genes) else np.ones(len(target_genes), dtype=np.float32) * 0.5
        else:
            reg_strength = np.ones(len(target_genes), dtype=np.float32) * 0.5
        
        # Activator or repressor
        is_activator = np.mean(binding_pattern) > 0
        
        return TranscriptionFactor(
            gene_id=mrna.gene_id,
            gene_type=GeneType.REGULATORY,
            strength=mrna.expression_level,
            binding_pattern=binding_pattern,
            target_genes=target_genes,
            regulation_strength=reg_strength,
            is_activator=is_activator,
        )
    
    def _translate_receptor(self, mrna: mRNA, gene: GeneEntry) -> ReceptorProtein:
        """mRNA → receptor protein."""
        vals = mrna.values
        d = gene.d_in if gene.d_in > 0 else 64
        
        selectivity = np.zeros(d, dtype=np.float32)
        selectivity[:min(d, len(vals))] = vals[:min(d, len(vals))]
        
        sensitivity = float(1.0 / (1.0 + np.exp(-vals[min(d, len(vals)-1)]))) if len(vals) > d else 0.5
        excitatory = float(np.mean(vals[:min(4, len(vals))])) > 0
        
        return ReceptorProtein(
            gene_id=mrna.gene_id,
            gene_type=GeneType.RECEPTOR,
            strength=mrna.expression_level,
            selectivity=selectivity,
            sensitivity=sensitivity,
            excitatory=excitatory,
        )
    
    def _translate_channel(self, mrna: mRNA, gene: GeneEntry) -> ChannelProtein:
        """mRNA → ion channel protein."""
        vals = mrna.values
        
        threshold = float(1.0 / (1.0 + np.exp(-vals[0]))) if len(vals) > 0 else 0.5
        conductance = float(1.0 / (1.0 + np.exp(-vals[1]))) if len(vals) > 1 else 0.5
        refractory = float(np.log(1 + np.exp(vals[2]))) if len(vals) > 2 else 1.0
        
        gate_size = min(64, max(0, len(vals) - 3))
        gate_weights = vals[3:3 + gate_size] if gate_size > 0 else None
        
        return ChannelProtein(
            gene_id=mrna.gene_id,
            gene_type=GeneType.CHANNEL,
            strength=mrna.expression_level,
            threshold=threshold,
            conductance=conductance,
            refractory_period=refractory,
            gate_weights=gate_weights,
        )
    
    def _translate_signal(self, mrna: mRNA, gene: GeneEntry) -> SignalProtein:
        """mRNA → signaling protein."""
        vals = mrna.values
        d_in, d_out = gene.d_in, gene.d_out
        
        W_size = d_in * d_out
        if d_in > 0 and d_out > 0 and len(vals) >= W_size:
            encoding_W = vals[:W_size].reshape(d_in, d_out)
        else:
            encoding_W = None
        
        signal_type = int(vals[-1] * 10) % 8 if len(vals) > 0 else 0
        
        return SignalProtein(
            gene_id=mrna.gene_id,
            gene_type=GeneType.SIGNAL,
            strength=mrna.expression_level,
            encoding_W=encoding_W,
            signal_type=signal_type,
        )
    
    def _translate_metabolic(self, mrna: mRNA, gene: GeneEntry) -> Enzyme:
        """mRNA → enzyme."""
        vals = mrna.values
        d = gene.d_in if gene.d_in > 0 else 64
        
        W_size = d * d
        if len(vals) >= W_size + d:
            transform = vals[:W_size].reshape(d, d)
            target_state = vals[W_size:W_size + d]
        elif len(vals) >= W_size:
            transform = vals[:W_size].reshape(d, d)
            target_state = np.zeros(d, dtype=np.float32)
        else:
            transform = None
            target_state = np.zeros(d, dtype=np.float32)
        
        return Enzyme(
            gene_id=mrna.gene_id,
            gene_type=GeneType.METABOLIC,
            strength=mrna.expression_level,
            transform=transform,
            target_state=target_state,
        )
    
    def _translate_modulatory(self, mrna: mRNA, gene: GeneEntry) -> Neuromodulator:
        """mRNA → neuromodulator."""
        vals = mrna.values
        
        pattern = np.zeros(64, dtype=np.float32)
        pattern[:min(64, len(vals))] = vals[:min(64, len(vals))]
        
        diffusion = float(1.0 / (1.0 + np.exp(-vals[64]))) if len(vals) > 64 else 0.5
        
        effect_size = min(65, max(0, len(vals) - 65))
        expression_effect = vals[65:65 + effect_size] if effect_size > 0 else None
        
        threshold_effect = float(vals[-1]) if len(vals) > 66 else 0.0
        
        return Neuromodulator(
            gene_id=mrna.gene_id,
            gene_type=GeneType.MODULATORY,
            strength=mrna.expression_level,
            pattern=pattern,
            diffusion_range=diffusion,
            expression_effect=expression_effect,
            threshold_effect=threshold_effect,
        )
    
    # ---- FULL PIPELINE ----
    
    def express_gene(self, gene_id: int, 
                     active_tfs: List[TranscriptionFactor],
                     context: Optional[np.ndarray] = None,
                     activity_level: float = 0.0) -> Optional[Protein]:
        """
        Full Central Dogma pipeline for one gene.
        
        DNA → accessibility check → TF binding → transcription → 
        splicing → translation → protein
        
        Args:
            gene_id: which gene to express
            active_tfs: list of active transcription factors in this cell
            context: current neuron state (for context-dependent splicing)
            activity_level: how active the neuron is (affects expression)
        
        Returns:
            Protein object, or None if gene is silenced/inactive
        """
        # Step 1: Accessibility
        if not self.is_accessible(gene_id):
            return None
        
        gene = self.genome.get_gene(gene_id)
        
        # Step 2: TF binding → expression level
        tf_activation = self.compute_tf_activation(gene_id, active_tfs)
        
        # Activity-dependent modulation
        tf_activation += activity_level * gene.activity_sensitivity
        
        # Sigmoid → expression level
        expression_level = 1.0 / (1.0 + np.exp(-tf_activation))
        
        # Step 3: Transcription
        mrna = self.transcribe(gene_id, expression_level, context)
        if mrna is None:
            return None
        
        # Step 5: Translation
        protein = self.translate(mrna)
        
        return protein
    
    def express_all_accessible(self, active_tfs: List[TranscriptionFactor],
                                context: Optional[np.ndarray] = None,
                                activity_level: float = 0.0) -> Dict[int, Protein]:
        """
        Express all accessible genes. Returns gene_id → protein mapping.
        """
        proteins = {}
        for gene_id in self.genome.get_accessible_genes(self.ACCESS_THRESHOLD):
            protein = self.express_gene(gene_id, active_tfs, context, activity_level)
            if protein is not None:
                proteins[gene_id] = protein
        return proteins
    
    def express_chromosome(self, chr_id: int,
                           active_tfs: List[TranscriptionFactor],
                           context: Optional[np.ndarray] = None,
                           activity_level: float = 0.0) -> Dict[int, Protein]:
        """Express all accessible genes in a specific chromosome."""
        proteins = {}
        for gene_id in self.genome.get_genes_by_chromosome(chr_id):
            protein = self.express_gene(gene_id, active_tfs, context, activity_level)
            if protein is not None:
                proteins[gene_id] = protein
        return proteins


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":
    import tempfile
    import os
    import time
    from genome import create_minimal_genome
    
    print("=" * 70)
    print("V7 Central Dogma — DNA → RNA → Protein Pipeline")
    print("=" * 70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create genome
        path = os.path.join(tmpdir, "test.genome")
        genome = create_minimal_genome(d_model=64, path=path)
        engine = TranscriptionEngine(genome)
        
        print(f"\nGenome: {genome.n_genes} genes, {genome.n_chromosomes} chromosomes")
        
        # Test 1: Express all genes with no TFs (base expression only)
        print("\n--- Expression with no TFs (base expression only) ---")
        proteins = engine.express_all_accessible(active_tfs=[])
        print(f"  Expressed {len(proteins)}/{genome.n_genes} genes")
        
        for gene_id, protein in sorted(proteins.items())[:10]:
            gene = genome.get_gene(gene_id)
            print(f"  Gene {gene_id} ({GeneType(gene.gene_type).name}): "
                  f"{type(protein).__name__} (strength={protein.strength:.3f})")
        
        # Test 2: Express structural gene → transform protein
        print("\n--- Structural Gene → Transform Protein ---")
        structural_ids = genome.get_genes_by_type(GeneType.STRUCTURAL)
        if structural_ids:
            gid = structural_ids[0]
            protein = engine.express_gene(gid, active_tfs=[])
            if isinstance(protein, TransformProtein):
                print(f"  Gene {gid}: W shape = {protein.W.shape}, b shape = {protein.b.shape}")
                x = np.random.randn(1, 64).astype(np.float32)
                y = protein(x)
                print(f"  Input: {x.shape} → Output: {y.shape}")
                print(f"  Output mean: {y.mean():.4f}, std: {y.std():.4f}")
        
        # Test 3: Express regulatory gene → transcription factor
        print("\n--- Regulatory Gene → Transcription Factor ---")
        reg_ids = genome.get_genes_by_type(GeneType.REGULATORY)
        if reg_ids:
            gid = reg_ids[0]
            protein = engine.express_gene(gid, active_tfs=[])
            if isinstance(protein, TranscriptionFactor):
                print(f"  Gene {gid}: binding_pattern shape = {protein.binding_pattern.shape}")
                print(f"  Target genes: {protein.target_genes}")
                print(f"  Is activator: {protein.is_activator}")
                
                # Test TF binding to another gene's promoter
                other_gene = 0
                promoter = genome.get_promoter(other_gene)
                affinity = protein.bind_affinity(promoter)
                print(f"  Binding affinity to gene {other_gene}: {affinity:.4f}")
        
        # Test 4: TF-regulated expression
        print("\n--- TF-Regulated Expression ---")
        # Create a TF, then use it to regulate expression
        tfs = []
        for gid in reg_ids[:3]:
            p = engine.express_gene(gid, active_tfs=[])
            if isinstance(p, TranscriptionFactor):
                tfs.append(p)
        
        if tfs:
            print(f"  Active TFs: {len(tfs)}")
            proteins_regulated = engine.express_all_accessible(active_tfs=tfs)
            print(f"  Expressed with TFs: {len(proteins_regulated)}/{genome.n_genes} genes")
            
            # Compare expression levels with/without TFs
            proteins_base = engine.express_all_accessible(active_tfs=[])
            for gid in sorted(set(proteins_base.keys()) & set(proteins_regulated.keys()))[:5]:
                base_str = proteins_base[gid].strength
                reg_str = proteins_regulated[gid].strength
                delta = reg_str - base_str
                print(f"  Gene {gid}: base={base_str:.3f} → regulated={reg_str:.3f} (Δ={delta:+.3f})")
        
        # Test 5: Context-dependent splicing
        print("\n--- Context-Dependent Splicing ---")
        structural_id = structural_ids[0] if structural_ids else 0
        
        ctx_calm = np.zeros(64, dtype=np.float32)
        ctx_active = np.random.randn(64).astype(np.float32) * 2.0
        
        p_calm = engine.express_gene(structural_id, [], context=ctx_calm)
        p_active = engine.express_gene(structural_id, [], context=ctx_active)
        
        if isinstance(p_calm, TransformProtein) and isinstance(p_active, TransformProtein):
            diff = np.abs(p_calm.W - p_active.W).mean()
            print(f"  Same gene, different contexts:")
            print(f"  Calm context → W mean: {p_calm.W.mean():.4f}, std: {p_calm.W.std():.4f}")
            print(f"  Active context → W mean: {p_active.W.mean():.4f}, std: {p_active.W.std():.4f}")
            print(f"  Mean |W difference|: {diff:.4f}")
        
        # Test 6: Receptor protein
        print("\n--- Receptor Protein ---")
        receptor_ids = genome.get_genes_by_type(GeneType.RECEPTOR)
        if receptor_ids:
            gid = receptor_ids[0]
            receptor = engine.express_gene(gid, active_tfs=[])
            if isinstance(receptor, ReceptorProtein):
                signal = np.random.randn(64).astype(np.float32)
                response = receptor.detect(signal)
                print(f"  Receptor gene {gid}: sensitivity={receptor.sensitivity:.3f}, "
                      f"excitatory={receptor.excitatory}")
                print(f"  Response to random signal: {response:.4f}")
        
        # Test 7: Channel protein
        print("\n--- Channel Protein ---")
        channel_ids = genome.get_genes_by_type(GeneType.CHANNEL)
        if channel_ids:
            gid = channel_ids[0]
            channel = engine.express_gene(gid, active_tfs=[])
            if isinstance(channel, ChannelProtein):
                print(f"  Channel gene {gid}: threshold={channel.threshold:.3f}, "
                      f"conductance={channel.conductance:.3f}, "
                      f"refractory={channel.refractory_period:.3f}")
                fire_low, str_low = channel.should_fire(0.3, time_since_fire=5.0)
                fire_high, str_high = channel.should_fire(0.8, time_since_fire=5.0)
                fire_ref, str_ref = channel.should_fire(0.8, time_since_fire=0.1)
                print(f"  Low potential (0.3): fire={fire_low}, strength={str_low:.3f}")
                print(f"  High potential (0.8): fire={fire_high}, strength={str_high:.3f}")
                print(f"  High + refractory: fire={fire_ref}, strength={str_ref:.3f}")
        
        # Test 8: Protein decay
        print("\n--- Protein Lifecycle ---")
        protein = engine.express_gene(structural_ids[0], active_tfs=[])
        if protein:
            initial_strength = protein.strength
            for step in range(100):
                protein.decay(rate=0.99)
            print(f"  After 100 steps (decay=0.99): {initial_strength:.3f} → {protein.strength:.3f}")
            print(f"  Still alive: {protein.is_alive}")
            
            for step in range(200):
                protein.decay(rate=0.99)
            print(f"  After 300 steps: strength={protein.strength:.6f}, alive={protein.is_alive}")
        
        # Test 9: Epigenetic silencing
        print("\n--- Epigenetic Silencing ---")
        gene_to_silence = structural_ids[1] if len(structural_ids) > 1 else 0
        
        p_before = engine.express_gene(gene_to_silence, active_tfs=[])
        print(f"  Gene {gene_to_silence} before silencing: {type(p_before).__name__ if p_before else 'None'}")
        
        genome.set_methylation(gene_to_silence, 0.9)  # silence
        p_after = engine.express_gene(gene_to_silence, active_tfs=[])
        print(f"  Gene {gene_to_silence} after methylation=0.9: {type(p_after).__name__ if p_after else 'SILENCED ✓'}")
        
        # Test 10: Pipeline benchmark
        print("\n--- Pipeline Benchmark ---")
        t0 = time.perf_counter()
        n_iterations = 1000
        for _ in range(n_iterations):
            proteins = engine.express_all_accessible(active_tfs=tfs[:1] if tfs else [])
        t1 = time.perf_counter()
        total_ms = (t1 - t0) * 1000
        per_iter_ms = total_ms / n_iterations
        print(f"  {n_iterations} full expression passes in {total_ms:.1f}ms")
        print(f"  {per_iter_ms:.3f}ms per pass ({len(proteins)} genes/pass)")
        print(f"  = {n_iterations * len(proteins) / (t1-t0):.0f} gene expressions/second")
        
        genome.close()
    
    print("\n✅ Central Dogma pipeline complete!")
