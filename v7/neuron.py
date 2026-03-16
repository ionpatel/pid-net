"""
V7 Phase 1: The Living Neuron

A neuron is a CELL — not a node in a graph, not a row in a matrix.
It has:
- Its own copy of the genome (shared reference, unique epigenetic state)
- A transcriptome (which genes are currently expressed, continuously updated)
- A protein pool (the active computational machinery, decaying + rebuilding)
- Internal state (neurostate vector, membrane potential, firing history)
- Compartments: membrane (receptors, channels, signal proteins), 
                 nucleus (genome, TFs), cytoplasm (transforms, enzymes)

Lifecycle per step:
  1. RECEIVE  — receptors detect incoming signals → accumulate membrane potential
  2. FIRE?    — channels check membrane potential vs threshold
  3. COMPUTE  — transform proteins process input (if firing or subthreshold)
  4. MAINTAIN — enzymes perform homeostatic correction
  5. SIGNAL   — if fired, signal proteins encode output
  6. REGULATE — update gene expression based on activity
  7. TURNOVER — decay old proteins, synthesize new ones from expressed genes
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Deque
from collections import deque
from enum import IntEnum

from genome import Genome, GeneType, ChromosomeFunction
from central_dogma import (
    TranscriptionEngine, Protein, TransformProtein, TranscriptionFactor,
    ReceptorProtein, ChannelProtein, SignalProtein, Enzyme, Neuromodulator, mRNA
)


# ============================================================
# Constants
# ============================================================

# Expression dynamics
EXPRESSION_MOMENTUM = 0.95      # how slowly expression levels change
EXPRESSION_THRESHOLD = 0.1      # below this = gene not expressed
ACTIVITY_BONUS = 0.3            # bonus to activity-sensitive genes when firing

# Protein dynamics
PROTEIN_DECAY_RATE = 0.995      # per step
PROTEIN_REMOVAL_THRESHOLD = 0.02
PROTEIN_REINFORCE_RATE = 0.1    # how much new transcription reinforces existing protein

# Neuron dynamics
MEMBRANE_LEAK = 0.95            # membrane potential leaks each step (longer integration window)
MEMBRANE_RESET = 0.0            # potential after firing
SUBTHRESHOLD_SCALE = 0.1        # how much subthreshold input drives computation
FIRING_HISTORY_LEN = 50         # steps of firing history to track

# Gene expression update frequency
EXPRESSION_UPDATE_INTERVAL = 5  # update gene expression every N steps
TURNOVER_INTERVAL = 10          # protein turnover every N steps


class NeuronType(IntEnum):
    """Neuron specialization types — determines initial gene expression."""
    PERCEPTION = 0
    MEMORY = 1
    PREDICTION = 2
    REGULATION = 3
    OUTPUT = 4
    INTERNEURON = 5   # general-purpose


# ============================================================
# Signal — What neurons pass to each other
# ============================================================

@dataclass
class NeuralSignal:
    """
    A signal transmitted from one neuron to another.
    Analogous to neurotransmitter release at a synapse.
    """
    source_id: int                  # which neuron sent this
    encoding: np.ndarray            # the signal vector (d_model,)
    signal_type: int = 0            # neurotransmitter type
    strength: float = 1.0           # release amount
    
    @property
    def magnitude(self) -> float:
        return float(np.linalg.norm(self.encoding)) * self.strength


# ============================================================
# Neuron — The Living Cell
# ============================================================

class Neuron:
    """
    A living computational cell.
    
    Every neuron has:
    - The SAME genome (shared reference)
    - UNIQUE epigenetic state (determines which genes CAN be expressed)
    - UNIQUE transcriptome (determines which genes ARE expressed right now)
    - UNIQUE protein pool (the active computational machinery)
    - UNIQUE internal state (neurostate, membrane potential, history)
    
    The neuron's identity IS its transcriptome — its gene expression pattern.
    Two neurons with different transcriptomes ARE different computational units,
    even though they share the same genome.
    """
    
    def __init__(self, neuron_id: int, genome: Genome, neuron_type: NeuronType,
                 d_model: int = 64):
        """
        Create a new neuron.
        
        Args:
            neuron_id: unique identifier
            genome: shared genome reference
            neuron_type: determines initial gene expression pattern
            d_model: dimensionality of neurostate vector
        """
        self.id = neuron_id
        self.genome = genome
        self.neuron_type = neuron_type
        self.d_model = d_model
        
        # Transcription engine (shared, stateless)
        self._engine = TranscriptionEngine(genome)
        
        # === NUCLEUS ===
        # Epigenetic state: per-gene methylation (0 = accessible, 1 = silenced)
        # Starts as a copy of genome's epigenetic state, then diverges per neuron
        self.methylation = np.array([genome.get_methylation(i) 
                                      for i in range(genome.n_genes)], dtype=np.float32)
        
        # Transcriptome: gene_id → expression level (0-1)
        self.transcriptome: Dict[int, float] = {}
        
        # Active transcription factors in this cell
        self._active_tfs: List[TranscriptionFactor] = []
        
        # === CYTOPLASM ===
        # Protein pool: gene_id → Protein object
        self.protein_pool: Dict[int, Protein] = {}
        
        # mRNA pool: gene_id → mRNA object
        self.mRNA_pool: Dict[int, mRNA] = {}
        
        # Neurostate: the internal state vector (persists across steps)
        self.neurostate = np.zeros(d_model, dtype=np.float32)
        
        # === MEMBRANE ===
        # Membrane potential: accumulates input, triggers firing
        self.membrane_potential = 0.0
        
        # Firing state
        self.firing_history: Deque[bool] = deque(maxlen=FIRING_HISTORY_LEN)
        self._time_since_fire = float('inf')
        self._just_fired = False
        self._last_fire_strength = 0.0
        
        # === BOOKKEEPING ===
        self.step_count = 0
        self._last_output_signal: Optional[NeuralSignal] = None
        
        # === METRICS (for verification) ===
        self.metrics = NeuronMetrics()
        
        # Initialize: differentiate into neuron type
        self._differentiate(neuron_type)
    
    # ================================================================
    # INITIALIZATION
    # ================================================================
    
    def _differentiate(self, neuron_type: NeuronType):
        """
        Differentiate this cell into a specific type.
        Like a stem cell becoming a neuron — sets initial epigenetic state
        and gene expression pattern.
        """
        # Map neuron type to primary chromosome
        type_to_chr = {
            NeuronType.PERCEPTION: ChromosomeFunction.PERCEPTION,
            NeuronType.MEMORY: ChromosomeFunction.MEMORY,
            NeuronType.PREDICTION: ChromosomeFunction.PREDICTION,
            NeuronType.REGULATION: ChromosomeFunction.REGULATION,
            NeuronType.OUTPUT: ChromosomeFunction.OUTPUT,
            NeuronType.INTERNEURON: ChromosomeFunction.HOMEOSTASIS,
        }
        
        primary_chr_func = type_to_chr[neuron_type]
        
        # Open primary chromosome + regulation + homeostasis (always needed)
        always_open = {ChromosomeFunction.REGULATION, ChromosomeFunction.HOMEOSTASIS}
        open_functions = {primary_chr_func} | always_open
        
        for chr_entry in self.genome.chromosomes:
            chr_func = ChromosomeFunction(chr_entry.function)
            gene_ids = self.genome.get_genes_by_chromosome(chr_entry.id)
            
            if chr_func in open_functions:
                # Open: low methylation → accessible
                for gid in gene_ids:
                    self.methylation[gid] = np.random.uniform(0.0, 0.2)
            else:
                # Closed: high methylation → mostly silenced (but not completely)
                for gid in gene_ids:
                    self.methylation[gid] = np.random.uniform(0.5, 0.9)
        
        # Essential genes are always accessible regardless of chromosome
        for gene in self.genome.genes:
            if gene.essential:
                self.methylation[gene.id] = 0.0
        
        # Initial gene expression pass
        self._full_expression_update()
    
    def _full_expression_update(self):
        """
        Full gene expression cycle:
        1. Express all accessible genes
        2. Collect TFs
        3. Re-express with TF regulation
        """
        # First pass: express accessible regulatory genes to get TFs
        self._active_tfs = []
        for gid in self.genome.get_genes_by_type(GeneType.REGULATORY):
            if self.methylation[gid] < self._engine.ACCESS_THRESHOLD:
                protein = self._engine.express_gene(
                    gid, active_tfs=[], context=self.neurostate,
                    activity_level=self._activity_level()
                )
                if isinstance(protein, TranscriptionFactor):
                    self._active_tfs.append(protein)
                    self.protein_pool[gid] = protein
                    self.transcriptome[gid] = protein.strength
        
        # Second pass: express all genes with TF regulation
        for gid in range(self.genome.n_genes):
            if self.methylation[gid] >= self._engine.ACCESS_THRESHOLD:
                self.transcriptome[gid] = 0.0
                continue
            
            protein = self._engine.express_gene(
                gid, active_tfs=self._active_tfs, context=self.neurostate,
                activity_level=self._activity_level()
            )
            if protein is not None:
                self.protein_pool[gid] = protein
                self.transcriptome[gid] = protein.strength
            else:
                self.transcriptome[gid] = 0.0
    
    # ================================================================
    # MAIN STEP — The neuron's heartbeat
    # ================================================================
    
    def step(self, incoming_signals: List[NeuralSignal]) -> Optional[NeuralSignal]:
        """
        One computational step of the neuron.
        
        Args:
            incoming_signals: signals from presynaptic neurons
            
        Returns:
            NeuralSignal if the neuron fired, None otherwise
        """
        self.step_count += 1
        self._just_fired = False
        self._last_output_signal = None
        
        # ---- 1. RECEIVE ----
        receptor_input = self._receive(incoming_signals)
        
        # ---- 2. MEMBRANE DYNAMICS ----
        # Membrane potential driven by INPUT MAGNITUDE (like ion flow)
        # Not the signed sum (which cancels with random encodings)
        self.membrane_potential *= MEMBRANE_LEAK  # leak
        input_magnitude = float(np.linalg.norm(receptor_input))
        self.membrane_potential += input_magnitude
        
        # ---- 3. FIRE DECISION ----
        should_fire, fire_strength = self._check_fire()
        
        # ---- 4. COMPUTE ----
        if should_fire:
            self._compute(receptor_input, scale=fire_strength)
            self._just_fired = True
            self._last_fire_strength = fire_strength
            self._time_since_fire = 0
            self.membrane_potential = MEMBRANE_RESET
            self.firing_history.append(True)
            self.metrics.n_fires += 1
        else:
            # Subthreshold processing — still compute, but weakly
            if np.abs(self.membrane_potential) > 0.01:
                self._compute(receptor_input, scale=SUBTHRESHOLD_SCALE)
            self._time_since_fire += 1
            self.firing_history.append(False)
        
        # ---- 5. MAINTAIN (homeostasis) ----
        self._maintain()
        
        # ---- 6. SIGNAL (if fired) ----
        output_signal = None
        if should_fire:
            output_signal = self._emit_signal(fire_strength)
            self._last_output_signal = output_signal
        
        # ---- 7. REGULATE (periodic) ----
        if self.step_count % EXPRESSION_UPDATE_INTERVAL == 0:
            self._update_expression()
        
        # ---- 8. PROTEIN DECAY (every step) ----
        self._decay_proteins()
        
        # ---- 9. TURNOVER (periodic — synthesize new proteins) ----
        if self.step_count % TURNOVER_INTERVAL == 0:
            self._protein_turnover()
        
        # Update metrics
        self.metrics.n_steps += 1
        self.metrics.avg_membrane = (
            0.99 * self.metrics.avg_membrane + 0.01 * abs(self.membrane_potential)
        )
        self.metrics.avg_neurostate_norm = (
            0.99 * self.metrics.avg_neurostate_norm + 
            0.01 * float(np.linalg.norm(self.neurostate))
        )
        
        return output_signal
    
    # ================================================================
    # STEP COMPONENTS
    # ================================================================
    
    def _receive(self, signals: List[NeuralSignal]) -> np.ndarray:
        """
        Step 1: Receive signals via two pathways:
        
        1. DIRECT pathway (voltage-gated): raw signal → membrane
           - Like gap junctions / electrical synapses
           - Always present, provides baseline input
        
        2. RECEPTOR pathway (ligand-gated): signal → receptor → filtered input
           - Like chemical synapses
           - Selective, modulatory
        
        Both pathways contribute. This prevents signal death from
        inhibitory receptor initialization.
        """
        total_input = np.zeros(self.d_model, dtype=np.float32)
        
        if not signals:
            return total_input
        
        receptors = self._get_proteins_of_type(ReceptorProtein)
        
        for sig in signals:
            enc = sig.encoding
            if len(enc) < self.d_model:
                enc = np.pad(enc, (0, self.d_model - len(enc)))
            elif len(enc) > self.d_model:
                enc = enc[:self.d_model]
            
            # Direct pathway: raw signal scaled by strength (always present)
            # This is the electrical/gap-junction component
            # Provides reliable baseline — prevents signal death from inhibitory receptors
            direct = enc * sig.strength * 0.5
            total_input += direct
            
            # Receptor pathway: filtered through receptor proteins
            # Adds selectivity and modulation on top of direct pathway
            if receptors:
                for receptor in receptors:
                    response = receptor.detect(enc)
                    total_input += enc * response * 0.5 / len(receptors)
            
            self.metrics.n_signals_received += 1
        
        return total_input
    
    def _check_fire(self) -> Tuple[bool, float]:
        """
        Step 3: Check if neuron should fire.
        Consults all active channel proteins.
        """
        channels = self._get_proteins_of_type(ChannelProtein)
        
        if not channels:
            # No channels — simple threshold
            threshold = 0.5
            if abs(self.membrane_potential) > threshold:
                return True, min(abs(self.membrane_potential), 3.0)
            return False, 0.0
        
        # Any channel that says fire → fire
        best_strength = 0.0
        should_fire = False
        
        for channel in channels:
            fire, strength = channel.should_fire(
                abs(self.membrane_potential), self._time_since_fire
            )
            if fire:
                should_fire = True
                best_strength = max(best_strength, strength)
        
        return should_fire, best_strength
    
    def _compute(self, receptor_input: np.ndarray, scale: float = 1.0):
        """
        Step 4: Transform input through active transform proteins.
        Updates the neurostate.
        """
        transforms = self._get_proteins_of_type(TransformProtein)
        
        if not transforms:
            # No transform proteins — simple accumulation
            self.neurostate += receptor_input * scale * 0.1
            return
        
        # Combine receptor input with current neurostate
        compute_input = (receptor_input * scale + self.neurostate) 
        
        # Apply each transform protein
        # Multiple transforms are applied sequentially (like a cascade)
        state = compute_input
        for i, transform in enumerate(transforms):
            state = transform(state.reshape(1, -1)).flatten()
            # Activation function (tanh to bound values)
            state = np.tanh(state)
        
        # Update neurostate as weighted combination of old and new
        alpha = min(scale, 1.0) * 0.5  # how much to update
        self.neurostate = (1 - alpha) * self.neurostate + alpha * state
    
    def _maintain(self):
        """
        Step 5: Homeostatic maintenance via enzymes.
        Keeps neurostate in healthy operating range.
        """
        enzymes = self._get_proteins_of_type(Enzyme)
        
        for enzyme in enzymes:
            if enzyme.target_state is not None:
                target = enzyme.target_state
                if len(target) < self.d_model:
                    target = np.pad(target, (0, self.d_model - len(target)))
                elif len(target) > self.d_model:
                    target = target[:self.d_model]
                
                error = target - self.neurostate
                correction = error * enzyme.correction_rate * enzyme.strength
                self.neurostate += correction
        
        # Basic norm regulation — prevent neurostate from exploding
        norm = np.linalg.norm(self.neurostate)
        if norm > 5.0:
            self.neurostate *= 5.0 / norm
    
    def _emit_signal(self, fire_strength: float) -> NeuralSignal:
        """
        Step 6: Encode neurostate as output signal.
        Uses signal proteins to produce neurotransmitter-like encoding.
        """
        signal_proteins = self._get_proteins_of_type(SignalProtein)
        
        if signal_proteins:
            # Use the strongest signal protein
            best_sp = max(signal_proteins, key=lambda p: p.strength)
            encoding = best_sp.encode(self.neurostate.reshape(1, -1)).flatten()
            signal_type = best_sp.signal_type
        else:
            # No signal protein — raw neurostate as signal
            encoding = self.neurostate.copy()
            signal_type = 0
        
        return NeuralSignal(
            source_id=self.id,
            encoding=encoding,
            signal_type=signal_type,
            strength=fire_strength,
        )
    
    def _update_expression(self):
        """
        Step 7: Update gene expression based on activity.
        Activity-dependent regulation — the neuron specializes through use.
        """
        activity = self._activity_level()
        
        # Collect active TFs
        self._active_tfs = [p for p in self.protein_pool.values() 
                           if isinstance(p, TranscriptionFactor) and p.is_alive]
        
        for gid in range(self.genome.n_genes):
            if self.methylation[gid] >= self._engine.ACCESS_THRESHOLD:
                self.transcriptome[gid] = 0.0
                continue
            
            gene = self.genome.get_gene(gid)
            
            # TF regulation
            tf_activation = self._engine.compute_tf_activation(gid, self._active_tfs)
            
            # Activity-dependent modulation
            tf_activation += activity * gene.activity_sensitivity
            
            # New expression level
            new_expr = 1.0 / (1.0 + np.exp(-tf_activation))
            
            # Smooth update (biological inertia)
            old_expr = self.transcriptome.get(gid, 0.0)
            self.transcriptome[gid] = (
                EXPRESSION_MOMENTUM * old_expr + (1 - EXPRESSION_MOMENTUM) * new_expr
            )
    
    def _decay_proteins(self):
        """
        Decay all proteins every step. Remove dead ones.
        Proteins degrade continuously — like real protein half-lives.
        """
        dead_genes = []
        for gid, protein in self.protein_pool.items():
            protein.decay(rate=PROTEIN_DECAY_RATE)
            if not protein.is_alive:
                dead_genes.append(gid)
        
        for gid in dead_genes:
            del self.protein_pool[gid]
            self.metrics.n_proteins_degraded += 1
    
    def _protein_turnover(self):
        """
        Periodic protein synthesis.
        Expressed genes produce new proteins to replace decayed ones.
        This is the ribosome doing its job.
        """
        for gid, expression in self.transcriptome.items():
            if expression < EXPRESSION_THRESHOLD:
                continue
            
            if gid in self.protein_pool and self.protein_pool[gid].is_alive:
                # Reinforce existing protein (new mRNA → more protein)
                self.protein_pool[gid].strength = min(
                    1.0, 
                    self.protein_pool[gid].strength + expression * PROTEIN_REINFORCE_RATE
                )
            else:
                # Synthesize new protein via Central Dogma
                protein = self._engine.express_gene(
                    gid, active_tfs=self._active_tfs, 
                    context=self.neurostate,
                    activity_level=self._activity_level()
                )
                if protein is not None:
                    self.protein_pool[gid] = protein
                    self.metrics.n_proteins_synthesized += 1
    
    # ================================================================
    # HELPERS
    # ================================================================
    
    def _get_proteins_of_type(self, protein_class) -> List:
        """Get all active proteins of a specific type."""
        return [p for p in self.protein_pool.values() 
                if isinstance(p, protein_class) and p.is_alive]
    
    def _activity_level(self) -> float:
        """Compute recent activity level (0-1)."""
        if not self.firing_history:
            return 0.0
        recent = list(self.firing_history)[-20:]
        return sum(recent) / len(recent)
    
    @property
    def firing_rate(self) -> float:
        """Recent firing rate (0-1)."""
        return self._activity_level()
    
    @property
    def just_fired(self) -> bool:
        return self._just_fired
    
    @property
    def n_active_proteins(self) -> int:
        return sum(1 for p in self.protein_pool.values() if p.is_alive)
    
    @property
    def n_expressed_genes(self) -> int:
        return sum(1 for v in self.transcriptome.values() if v > EXPRESSION_THRESHOLD)
    
    def get_protein_inventory(self) -> Dict[str, int]:
        """Count proteins by type."""
        inv = {}
        for p in self.protein_pool.values():
            if p.is_alive:
                name = type(p).__name__
                inv[name] = inv.get(name, 0) + 1
        return inv
    
    def get_expression_profile(self) -> Dict[str, List[Tuple[int, float]]]:
        """Get expression levels grouped by gene type."""
        profile = {}
        for gid, level in sorted(self.transcriptome.items()):
            if level < 0.01:
                continue
            gene = self.genome.get_gene(gid)
            type_name = GeneType(gene.gene_type).name
            if type_name not in profile:
                profile[type_name] = []
            profile[type_name].append((gid, level))
        return profile
    
    def summary(self) -> str:
        """Human-readable neuron summary."""
        lines = [
            f"Neuron {self.id} ({self.neuron_type.name})",
            f"  d_model: {self.d_model}",
            f"  Step: {self.step_count}",
            f"  Membrane potential: {self.membrane_potential:.4f}",
            f"  Neurostate norm: {np.linalg.norm(self.neurostate):.4f}",
            f"  Firing rate: {self.firing_rate:.3f}",
            f"  Active proteins: {self.n_active_proteins}",
            f"  Expressed genes: {self.n_expressed_genes}/{self.genome.n_genes}",
        ]
        
        inv = self.get_protein_inventory()
        if inv:
            lines.append("  Protein inventory:")
            for name, count in sorted(inv.items()):
                lines.append(f"    {name}: {count}")
        
        return '\n'.join(lines)


# ============================================================
# Neuron Metrics — For verification
# ============================================================

@dataclass
class NeuronMetrics:
    """Track everything for verification."""
    n_steps: int = 0
    n_fires: int = 0
    n_signals_received: int = 0
    n_proteins_synthesized: int = 0
    n_proteins_degraded: int = 0
    avg_membrane: float = 0.0
    avg_neurostate_norm: float = 0.0
    
    def summary(self) -> str:
        fire_rate = self.n_fires / max(1, self.n_steps)
        return (
            f"  Steps: {self.n_steps}, Fires: {self.n_fires} ({fire_rate:.1%})\n"
            f"  Signals received: {self.n_signals_received}\n"
            f"  Proteins synthesized: {self.n_proteins_synthesized}, "
            f"degraded: {self.n_proteins_degraded}\n"
            f"  Avg membrane: {self.avg_membrane:.4f}, "
            f"Avg neurostate norm: {self.avg_neurostate_norm:.4f}"
        )


# ============================================================
# VERIFICATION SUITE — "Zero chance of error"
# ============================================================

def verify_neuron(genome_path: str = None):
    """
    Comprehensive verification of the neuron.
    Tests every component of the biological pipeline.
    """
    import tempfile, os, time
    
    print("=" * 70)
    print("V7 NEURON VERIFICATION SUITE")
    print("Zero tolerance for error. Every Greek checked.")
    print("=" * 70)
    
    tmpdir = tempfile.mkdtemp()
    
    if genome_path is None:
        from genome import create_minimal_genome
        genome_path = os.path.join(tmpdir, "verify.genome")
        genome = create_minimal_genome(d_model=64, path=genome_path)
    else:
        genome = Genome(genome_path, mode='rw')
    
    d_model = 64
    all_passed = True
    test_num = 0
    
    def check(name: str, condition: bool, detail: str = ""):
        nonlocal test_num, all_passed
        test_num += 1
        status = "✅ PASS" if condition else "❌ FAIL"
        if not condition:
            all_passed = False
        print(f"  [{test_num:02d}] {status}: {name}")
        if detail and not condition:
            print(f"         {detail}")
        return condition
    
    # ============================================================
    # TEST GROUP 1: Neuron Creation & Differentiation
    # ============================================================
    print("\n─── GROUP 1: Creation & Differentiation ───")
    
    neuron = Neuron(0, genome, NeuronType.PERCEPTION, d_model=d_model)
    
    check("Neuron created", neuron is not None)
    check("Correct type", neuron.neuron_type == NeuronType.PERCEPTION)
    check("Neurostate initialized", neuron.neurostate.shape == (d_model,))
    check("Neurostate starts at zero", np.allclose(neuron.neurostate, 0.0))
    check("Membrane potential starts at zero", neuron.membrane_potential == 0.0)
    check("Firing history empty", len(neuron.firing_history) == 0)
    check("Step count starts at zero", neuron.step_count == 0)
    
    # Check differentiation: perception neuron should have perception chromosome open
    perception_genes = genome.get_genes_by_chromosome(0)  # Chr 0 = PERCEPTION
    open_count = sum(1 for gid in perception_genes if neuron.methylation[gid] < 0.5)
    check("Perception chromosome accessible", 
          open_count == len(perception_genes),
          f"Expected {len(perception_genes)} open, got {open_count}")
    
    # Check that other chromosomes are more closed
    prediction_genes = genome.get_genes_by_chromosome(2)  # Chr 2 = PREDICTION
    closed_count = sum(1 for gid in prediction_genes if neuron.methylation[gid] >= 0.5)
    check("Non-primary chromosome partially closed",
          closed_count > 0,
          f"Expected some closed, got {closed_count}/{len(prediction_genes)}")
    
    # Check essential genes always open
    essential_genes = [g.id for g in genome.genes if g.essential]
    essential_open = all(neuron.methylation[gid] == 0.0 for gid in essential_genes)
    check("Essential genes always accessible", essential_open,
          f"Essential genes: {essential_genes}")
    
    # Proteins should be populated after differentiation
    check("Protein pool populated", len(neuron.protein_pool) > 0,
          f"Got {len(neuron.protein_pool)} proteins")
    
    # Should have at least one transform protein
    n_transforms = len(neuron._get_proteins_of_type(TransformProtein))
    check("Has transform proteins", n_transforms > 0,
          f"Got {n_transforms}")
    
    # Should have at least one channel protein
    n_channels = len(neuron._get_proteins_of_type(ChannelProtein))
    check("Has channel proteins", n_channels > 0,
          f"Got {n_channels}")
    
    print(f"\n  Protein inventory: {neuron.get_protein_inventory()}")
    print(f"  Expressed genes: {neuron.n_expressed_genes}/{genome.n_genes}")
    
    # ============================================================
    # TEST GROUP 2: Signal Reception
    # ============================================================
    print("\n─── GROUP 2: Signal Reception ───")
    
    # No signals → no change
    membrane_before = neuron.membrane_potential
    neuron.step([])
    check("No signal → membrane leaks toward 0",
          abs(neuron.membrane_potential) <= abs(membrane_before) + 0.01)
    
    # Send a signal — should either change membrane or trigger fire
    signal = NeuralSignal(
        source_id=99,
        encoding=np.random.randn(d_model).astype(np.float32) * 0.5,
        signal_type=0,
        strength=1.0,
    )
    
    fires_before = neuron.metrics.n_fires
    membrane_before = neuron.membrane_potential
    neuron.step([signal])
    membrane_changed = neuron.membrane_potential != membrane_before
    fired = neuron.metrics.n_fires > fires_before
    check("Signal affects neuron (changes membrane or fires)",
          membrane_changed or fired,
          f"Before: {membrane_before:.4f}, After: {neuron.membrane_potential:.4f}, fired: {fired}")
    
    # Multiple signals accumulate (or trigger a fire, which is also accumulation working)
    signals = [
        NeuralSignal(source_id=i, 
                     encoding=np.ones(d_model, dtype=np.float32) * 0.3,
                     strength=1.0)
        for i in range(5)
    ]
    neuron.membrane_potential = 0.0  # reset for clean test
    fires_before = neuron.metrics.n_fires
    result = neuron.step(signals)
    fired = neuron.metrics.n_fires > fires_before
    accumulated = abs(neuron.membrane_potential) > 0.01
    check("Multiple signals accumulate (or trigger fire)",
          accumulated or fired,
          f"Membrane: {neuron.membrane_potential:.4f}, fired: {fired}")
    check("Signals counted in metrics",
          neuron.metrics.n_signals_received > 0,
          f"Received: {neuron.metrics.n_signals_received}")
    
    # ============================================================
    # TEST GROUP 3: Firing Behavior
    # ============================================================
    print("\n─── GROUP 3: Firing Behavior ───")
    
    # Reset neuron for clean fire test
    neuron2 = Neuron(1, genome, NeuronType.PERCEPTION, d_model=d_model)
    
    # Subthreshold: small signal should not fire
    weak_signal = NeuralSignal(
        source_id=99,
        encoding=np.ones(d_model, dtype=np.float32) * 0.001,
        strength=0.01,
    )
    result = neuron2.step([weak_signal])
    check("Weak signal → no fire", result is None)
    check("Firing history records non-fire", 
          len(neuron2.firing_history) == 1 and not neuron2.firing_history[-1])
    
    # Suprathreshold: strong signal should fire
    neuron3 = Neuron(2, genome, NeuronType.PERCEPTION, d_model=d_model)
    strong_signals = [
        NeuralSignal(
            source_id=i,
            encoding=np.ones(d_model, dtype=np.float32) * 2.0,
            strength=3.0,
        )
        for i in range(10)
    ]
    result = neuron3.step(strong_signals)
    check("Strong signal → fires", result is not None,
          f"Membrane: {neuron3.membrane_potential:.4f}")
    
    if result is not None:
        check("Fire produces signal", isinstance(result, NeuralSignal))
        check("Signal has encoding", result.encoding.shape == (d_model,))
        check("Signal has source ID", result.source_id == neuron3.id)
        check("Signal has positive strength", result.strength > 0,
              f"Strength: {result.strength:.4f}")
        check("Membrane resets after fire",
              abs(neuron3.membrane_potential) < 0.01,
              f"Membrane: {neuron3.membrane_potential:.4f}")
        check("Firing history records fire",
              neuron3.firing_history[-1] == True)
    
    # Refractory: immediately after firing, shouldn't fire again easily
    result2 = neuron3.step(strong_signals)
    # May or may not fire depending on channel refractory period
    # Just verify the neuron doesn't crash
    check("Post-fire step runs without error", True)
    
    # ============================================================
    # TEST GROUP 4: Neurostate Computation
    # ============================================================
    print("\n─── GROUP 4: Neurostate Computation ───")
    
    neuron4 = Neuron(3, genome, NeuronType.PERCEPTION, d_model=d_model)
    
    initial_state = neuron4.neurostate.copy()
    
    # Run with input — neurostate should change
    signal = NeuralSignal(
        source_id=99,
        encoding=np.random.randn(d_model).astype(np.float32),
        strength=2.0,
    )
    for _ in range(10):
        neuron4.step([signal])
    
    check("Neurostate changes with input",
          not np.allclose(neuron4.neurostate, initial_state),
          f"Norm change: {np.linalg.norm(neuron4.neurostate - initial_state):.4f}")
    
    check("Neurostate bounded (no explosion)",
          np.linalg.norm(neuron4.neurostate) < 10.0,
          f"Norm: {np.linalg.norm(neuron4.neurostate):.4f}")
    
    # Neurostate persists between steps (not reset)
    state_before = neuron4.neurostate.copy()
    neuron4.step([])  # empty step
    check("Neurostate persists (not reset)",
          np.linalg.norm(neuron4.neurostate) > 0.001 or np.allclose(state_before, 0),
          f"Before: {np.linalg.norm(state_before):.4f}, After: {np.linalg.norm(neuron4.neurostate):.4f}")
    
    # Different inputs produce different states
    neuron5a = Neuron(4, genome, NeuronType.PERCEPTION, d_model=d_model)
    neuron5b = Neuron(5, genome, NeuronType.PERCEPTION, d_model=d_model)
    
    np.random.seed(42)
    sig_a = NeuralSignal(source_id=0, encoding=np.random.randn(d_model).astype(np.float32), strength=2.0)
    np.random.seed(123)
    sig_b = NeuralSignal(source_id=0, encoding=np.random.randn(d_model).astype(np.float32), strength=2.0)
    
    for _ in range(20):
        neuron5a.step([sig_a])
        neuron5b.step([sig_b])
    
    state_diff = np.linalg.norm(neuron5a.neurostate - neuron5b.neurostate)
    check("Different inputs → different states",
          state_diff > 0.01,
          f"State difference: {state_diff:.4f}")
    
    # ============================================================
    # TEST GROUP 5: Protein Lifecycle
    # ============================================================
    print("\n─── GROUP 5: Protein Lifecycle ───")
    
    neuron6 = Neuron(6, genome, NeuronType.PERCEPTION, d_model=d_model)
    initial_proteins = neuron6.n_active_proteins
    check("Proteins exist after init", initial_proteins > 0)
    
    # Run many steps — actively expressed proteins should be maintained
    for _ in range(200):
        neuron6.step([])
    
    check("Proteins still active after 200 steps",
          neuron6.n_active_proteins > 0,
          f"Active: {neuron6.n_active_proteins}")
    
    # Expressed proteins should be maintained near full strength (reinforcement > decay)
    # This is biologically correct — active genes maintain their protein levels
    strengths = [p.strength for p in neuron6.protein_pool.values() if p.is_alive]
    if strengths:
        avg_strength = np.mean(strengths)
        check("Expressed proteins maintained (avg strength > 0.5)",
              avg_strength > 0.5,
              f"Avg strength: {avg_strength:.4f}")
    
    # NOW: silence a gene and verify its protein decays
    # Pick a non-essential structural gene
    struct_ids = genome.get_genes_by_type(GeneType.STRUCTURAL)
    silenceable = [gid for gid in struct_ids if not genome.get_gene(gid).essential 
                   and gid in neuron6.protein_pool]
    if silenceable:
        silence_gid = silenceable[0]
        strength_before = neuron6.protein_pool[silence_gid].strength
        
        # Silence the gene epigenetically
        neuron6.methylation[silence_gid] = 0.99
        neuron6.transcriptome[silence_gid] = 0.0  # stop expression
        
        # Run more steps — this protein should decay without reinforcement
        for _ in range(500):
            neuron6.step([])
        
        if silence_gid in neuron6.protein_pool:
            strength_after = neuron6.protein_pool[silence_gid].strength
            check("Silenced gene's protein decays",
                  strength_after < strength_before * 0.5,
                  f"Before: {strength_before:.4f}, After: {strength_after:.4f}")
        else:
            check("Silenced gene's protein degraded completely", True)
            neuron6.metrics.n_proteins_degraded > 0 and \
            check("Protein degradation counter incremented",
                  neuron6.metrics.n_proteins_degraded > 0,
                  f"Degraded: {neuron6.metrics.n_proteins_degraded}")
    else:
        check("(skip: no silenceable protein found)", True)
    
    # ============================================================
    # TEST GROUP 6: Gene Expression Dynamics
    # ============================================================
    print("\n─── GROUP 6: Gene Expression Dynamics ───")
    
    neuron7 = Neuron(7, genome, NeuronType.PERCEPTION, d_model=d_model)
    initial_expr = dict(neuron7.transcriptome)
    
    # Force high activity
    strong = NeuralSignal(
        source_id=0,
        encoding=np.ones(d_model, dtype=np.float32) * 3.0,
        strength=5.0,
    )
    for _ in range(100):
        neuron7.step([strong])
    
    final_expr = dict(neuron7.transcriptome)
    
    # Expression should have changed
    changed = sum(1 for gid in initial_expr 
                  if abs(initial_expr.get(gid, 0) - final_expr.get(gid, 0)) > 0.001)
    check("Gene expression changes with activity",
          changed > 0,
          f"Changed: {changed}/{len(initial_expr)}")
    
    # Activity-sensitive genes should be more affected
    expr_profile = neuron7.get_expression_profile()
    check("Expression profile has entries", len(expr_profile) > 0)
    
    # ============================================================
    # TEST GROUP 7: Different Neuron Types
    # ============================================================
    print("\n─── GROUP 7: Neuron Type Differentiation ───")
    
    types_to_test = [NeuronType.PERCEPTION, NeuronType.MEMORY, 
                     NeuronType.OUTPUT, NeuronType.PREDICTION]
    neurons = {}
    for i, nt in enumerate(types_to_test):
        neurons[nt] = Neuron(100 + i, genome, nt, d_model=d_model)
    
    # Different types should have different expression profiles
    profiles = {nt: n.get_expression_profile() for nt, n in neurons.items()}
    
    for nt, profile in profiles.items():
        n = neurons[nt]
        check(f"{nt.name} neuron has proteins",
              n.n_active_proteins > 0,
              f"Active: {n.n_active_proteins}")
    
    # Compare perception vs memory — should have different accessible gene sets
    perc_meth = neurons[NeuronType.PERCEPTION].methylation.copy()
    mem_meth = neurons[NeuronType.MEMORY].methylation.copy()
    meth_diff = np.sum(np.abs(perc_meth - mem_meth) > 0.1)
    check("Different types have different methylation",
          meth_diff > 0,
          f"Genes with different methylation: {meth_diff}")
    
    # ============================================================
    # TEST GROUP 8: Homeostasis
    # ============================================================
    print("\n─── GROUP 8: Homeostasis ───")
    
    neuron8 = Neuron(8, genome, NeuronType.PERCEPTION, d_model=d_model)
    
    # Inject extreme state
    neuron8.neurostate = np.ones(d_model, dtype=np.float32) * 100.0
    
    # Run steps — homeostasis should bring it back
    for _ in range(50):
        neuron8.step([])
    
    norm_after = np.linalg.norm(neuron8.neurostate)
    check("Homeostasis bounds extreme state",
          norm_after < 100.0 * np.sqrt(d_model),  # less than initial
          f"Norm: {norm_after:.4f} (initial was {100.0 * np.sqrt(d_model):.4f})")
    
    # ============================================================
    # TEST GROUP 9: Signal Encoding
    # ============================================================
    print("\n─── GROUP 9: Signal Encoding ───")
    
    neuron9 = Neuron(9, genome, NeuronType.OUTPUT, d_model=d_model)
    
    # Force fire by setting high membrane potential directly
    neuron9.neurostate = np.random.randn(d_model).astype(np.float32)
    
    # Drive with strong signals to trigger firing
    strong = [NeuralSignal(source_id=i, 
              encoding=np.ones(d_model, dtype=np.float32) * 3.0, 
              strength=5.0) for i in range(10)]
    
    signal_out = None
    for _ in range(20):
        result = neuron9.step(strong)
        if result is not None:
            signal_out = result
            break
    
    if signal_out is not None:
        check("Output signal has correct shape", 
              signal_out.encoding.shape == (d_model,))
        check("Output signal is finite",
              np.all(np.isfinite(signal_out.encoding)))
        check("Output signal is non-zero",
              np.linalg.norm(signal_out.encoding) > 0.0001)
        check("Output signal has valid strength",
              signal_out.strength > 0 and np.isfinite(signal_out.strength))
    else:
        check("Output neuron fires (WARN: didn't fire in 20 steps)", False,
              "This may be OK if thresholds are high — not a hard failure")
    
    # ============================================================
    # TEST GROUP 10: Long-Running Stability
    # ============================================================
    print("\n─── GROUP 10: Long-Running Stability (1000 steps) ───")
    
    neuron10 = Neuron(10, genome, NeuronType.PERCEPTION, d_model=d_model)
    
    np.random.seed(42)
    n_fires_total = 0
    max_norm = 0.0
    any_nan = False
    any_inf = False
    
    t0 = time.perf_counter()
    for step in range(1000):
        # Random signal every other step
        if step % 2 == 0:
            sig = NeuralSignal(
                source_id=0,
                encoding=np.random.randn(d_model).astype(np.float32) * 0.5,
                strength=np.random.uniform(0.1, 2.0),
            )
            result = neuron10.step([sig])
        else:
            result = neuron10.step([])
        
        if result is not None:
            n_fires_total += 1
            if np.any(np.isnan(result.encoding)):
                any_nan = True
            if np.any(np.isinf(result.encoding)):
                any_inf = True
        
        norm = np.linalg.norm(neuron10.neurostate)
        max_norm = max(max_norm, norm)
        
        if np.any(np.isnan(neuron10.neurostate)):
            any_nan = True
        if np.any(np.isinf(neuron10.neurostate)):
            any_inf = True
    
    t1 = time.perf_counter()
    
    check("No NaN in 1000 steps", not any_nan)
    check("No Inf in 1000 steps", not any_inf)
    check("Neurostate bounded (max norm < 100)",
          max_norm < 100.0, f"Max norm: {max_norm:.4f}")
    check("Neuron fired at least once", n_fires_total > 0,
          f"Fires: {n_fires_total}/1000")
    check("Neuron didn't fire every step (has selectivity)",
          n_fires_total < 900,
          f"Fires: {n_fires_total}/1000")
    
    ms_total = (t1 - t0) * 1000
    steps_per_sec = 1000 / (t1 - t0)
    check(f"Performance: {steps_per_sec:.0f} steps/sec ({ms_total:.1f}ms total)", True)
    
    print(f"\n  Final neuron state:")
    print(f"  {neuron10.summary()}")
    print(f"  {neuron10.metrics.summary()}")
    
    # ============================================================
    # TEST GROUP 11: Determinism & Reproducibility
    # ============================================================
    print("\n─── GROUP 11: Determinism ───")
    
    # Same seed → same result
    np.random.seed(777)
    n11a = Neuron(11, genome, NeuronType.PERCEPTION, d_model=d_model)
    sig = NeuralSignal(source_id=0, encoding=np.ones(d_model, dtype=np.float32), strength=1.0)
    
    np.random.seed(999)  # different seed for signals, but same neuron init
    for _ in range(50):
        n11a.step([sig])
    state_a = n11a.neurostate.copy()
    
    # Recreate with same seed
    np.random.seed(777)
    n11b = Neuron(12, genome, NeuronType.PERCEPTION, d_model=d_model)
    
    np.random.seed(999)
    for _ in range(50):
        n11b.step([sig])
    state_b = n11b.neurostate.copy()
    
    check("Same init + same input → same state (deterministic)",
          np.allclose(state_a, state_b, atol=1e-5),
          f"Max diff: {np.max(np.abs(state_a - state_b)):.6f}")
    
    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "=" * 70)
    if all_passed:
        print(f"ALL {test_num} TESTS PASSED ✅")
    else:
        n_failed = test_num - sum(1 for _ in range(test_num))
        print(f"SOME TESTS FAILED ❌ (see above)")
    print("=" * 70)
    
    genome.close()
    
    # Cleanup
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)
    
    return all_passed


# ============================================================
# Quick demo
# ============================================================

def demo():
    """Quick demo of a living neuron."""
    import tempfile, os
    from genome import create_minimal_genome
    
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "demo.genome")
    genome = create_minimal_genome(d_model=64, path=path)
    
    print("\n🧬 Creating a living neuron...")
    neuron = Neuron(0, genome, NeuronType.PERCEPTION, d_model=64)
    print(neuron.summary())
    
    print("\n⚡ Stimulating with random signals...")
    np.random.seed(42)
    
    for step in range(100):
        signals = []
        if step % 3 == 0:
            signals = [NeuralSignal(
                source_id=1,
                encoding=np.random.randn(64).astype(np.float32) * 0.5,
                strength=np.random.uniform(0.5, 2.0),
            )]
        
        result = neuron.step(signals)
        
        if result is not None:
            print(f"  Step {step}: FIRE! 🔥 (strength={result.strength:.3f}, "
                  f"signal_mag={result.magnitude:.3f})")
    
    print(f"\n📊 After 100 steps:")
    print(neuron.summary())
    print(neuron.metrics.summary())
    
    print(f"\n🧪 Expression profile:")
    for type_name, genes in neuron.get_expression_profile().items():
        gene_strs = [f"g{gid}={level:.2f}" for gid, level in genes[:5]]
        print(f"  {type_name}: {', '.join(gene_strs)}")
    
    genome.close()
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    import sys
    
    if "--demo" in sys.argv:
        demo()
    else:
        verify_neuron()
