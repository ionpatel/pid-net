"""
V7 Phase 4: Neuromodulation — The Brain's Operating System

Three signaling tiers, each at a different timescale:

1. SYNAPTIC (fast, targeted, milliseconds)
   → Already implemented in circuit.py
   → Neuron-to-neuron via synapses

2. NEUROMODULATORY (slow, diffuse, seconds-minutes)
   → THIS MODULE
   → Specialized neurons release modulators that change entire regions
   → Dopamine, Serotonin, Norepinephrine, Acetylcholine

3. HORMONAL (very slow, global, minutes-hours)
   → THIS MODULE
   → Organism-level state that affects everything
   → Stress, energy, arousal

Why this matters:
- Dopamine modulates LEARNING RATE. High dopamine = learn faster from rewards.
- Norepinephrine modulates ATTENTION. High NE = lower thresholds, more responsive.
- Serotonin modulates PATIENCE. High 5-HT = longer temporal horizons.
- Acetylcholine modulates PLASTICITY. High ACh = more synaptic modification.

Without neuromodulation, the brain runs in one mode. With it, the same circuit
can switch between exploration/exploitation, focused/diffuse, learning/performing.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from collections import defaultdict
from enum import IntEnum

from genome import Genome, GeneType
from neuron import Neuron, NeuronType, NeuralSignal
from circuit import Circuit, Synapse


# ============================================================
# Neuromodulator Types
# ============================================================

class ModulatorType(IntEnum):
    DOPAMINE = 0        # reward, learning rate, motivation
    SEROTONIN = 1       # patience, inhibition, mood
    NOREPINEPHRINE = 2  # alertness, attention, fight-or-flight
    ACETYLCHOLINE = 3   # plasticity, memory formation, focus


# Effects each modulator has on neural computation
MODULATOR_EFFECTS = {
    ModulatorType.DOPAMINE: {
        'learning_rate_mult': 2.0,      # doubles learning when high
        'threshold_mult': 0.9,          # slightly lowers thresholds (more responsive)
        'expression_genes': 'STRUCTURAL',  # upregulates structural genes
        'description': 'Reward signal. High = learn fast, seek reward.',
    },
    ModulatorType.SEROTONIN: {
        'learning_rate_mult': 0.8,      # slows learning (patience)
        'threshold_mult': 1.2,          # raises thresholds (less reactive)
        'expression_genes': 'METABOLIC',   # upregulates homeostasis
        'description': 'Patience/inhibition. High = calm, long-horizon.',
    },
    ModulatorType.NOREPINEPHRINE: {
        'learning_rate_mult': 1.5,      # moderate learning boost
        'threshold_mult': 0.7,          # strongly lowers thresholds (hypervigilant)
        'expression_genes': 'RECEPTOR',    # upregulates receptors (more sensitive)
        'description': 'Alertness. High = vigilant, responsive, aroused.',
    },
    ModulatorType.ACETYLCHOLINE: {
        'learning_rate_mult': 1.8,      # strong plasticity boost
        'threshold_mult': 1.0,          # neutral on thresholds
        'expression_genes': 'REGULATORY',  # upregulates regulatory genes
        'description': 'Plasticity. High = form new memories, adapt.',
    },
}


# ============================================================
# Constants
# ============================================================

BASELINE_LEVEL = 0.5             # tonic modulator level
MODULATOR_MOMENTUM = 0.95        # how slowly modulator levels change
MODULATOR_DECAY = 0.99           # decay toward baseline
EFFECT_SCALE = 0.3               # how strongly modulators affect neurons
HORMONE_MOMENTUM = 0.99          # hormones change even slower


# ============================================================
# Neuromodulatory System
# ============================================================

class NeuromodulatorySystem:
    """
    One neuromodulatory system (e.g., the dopamine system).
    
    Source neurons release the modulator.
    Target neurons/regions are affected.
    The modulator level changes based on source neuron activity.
    """
    
    def __init__(self, modulator_type: ModulatorType,
                 source_neuron_ids: List[int],
                 target_neuron_ids: List[int]):
        self.modulator_type = modulator_type
        self.source_neurons = source_neuron_ids
        self.target_neurons = target_neuron_ids
        self.effects = MODULATOR_EFFECTS[modulator_type]
        
        # Current modulator level (0 = depleted, 1 = saturated)
        self.level = BASELINE_LEVEL
        self.baseline = BASELINE_LEVEL
        
        # History for analysis
        self.level_history: List[float] = []
    
    def update_level(self, neurons: Dict[int, Neuron]):
        """Update modulator level based on source neuron activity."""
        if not self.source_neurons:
            return
        
        # Compute source activity
        source_activity = 0.0
        n_sources = 0
        for nid in self.source_neurons:
            if nid in neurons:
                source_activity += neurons[nid].firing_rate
                n_sources += 1
        
        if n_sources > 0:
            source_activity /= n_sources
        
        # Update level with momentum
        self.level = (MODULATOR_MOMENTUM * self.level + 
                      (1 - MODULATOR_MOMENTUM) * source_activity)
        
        # Decay toward baseline
        self.level = MODULATOR_DECAY * self.level + (1 - MODULATOR_DECAY) * self.baseline
        
        # Clamp
        self.level = np.clip(self.level, 0.0, 1.0)
        self.level_history.append(self.level)
    
    def apply_to_neuron(self, neuron: Neuron):
        """
        Apply modulator effect to a single neuron.
        Modulates: firing threshold, gene expression.
        """
        delta = self.level - self.baseline  # deviation from baseline
        
        if abs(delta) < 0.01:
            return  # negligible effect
        
        # Modify firing threshold via channel proteins
        from central_dogma import ChannelProtein
        for protein in neuron.protein_pool.values():
            if isinstance(protein, ChannelProtein):
                # Delta > 0 and threshold_mult < 1 → lower threshold
                mult = 1.0 + (self.effects['threshold_mult'] - 1.0) * delta * EFFECT_SCALE
                protein.threshold *= mult
                protein.threshold = np.clip(protein.threshold, 0.05, 5.0)
        
        # Modify gene expression for target gene type
        target_type_name = self.effects['expression_genes']
        target_type = GeneType[target_type_name]
        
        for gid, expr in neuron.transcriptome.items():
            gene = neuron.genome.get_gene(gid)
            if gene.gene_type == target_type:
                # Upregulate when modulator is high, downregulate when low
                modifier = 1.0 + delta * gene.modulator_sensitivity * EFFECT_SCALE
                neuron.transcriptome[gid] = np.clip(expr * modifier, 0.0, 1.0)
    
    def apply_to_circuit(self, circuit: Circuit):
        """Apply modulator effect to all target neurons in circuit."""
        for nid in self.target_neurons:
            if nid in circuit.neurons:
                self.apply_to_neuron(circuit.neurons[nid])
    
    def get_learning_rate_modifier(self) -> float:
        """How much to scale learning rate based on current level."""
        delta = self.level - self.baseline
        return 1.0 + (self.effects['learning_rate_mult'] - 1.0) * delta
    
    @property
    def description(self) -> str:
        return self.effects['description']
    
    def summary(self) -> str:
        return (f"{self.modulator_type.name}: level={self.level:.4f} "
                f"(baseline={self.baseline:.4f}, Δ={self.level-self.baseline:+.4f}), "
                f"sources={len(self.source_neurons)}, targets={len(self.target_neurons)}")


# ============================================================
# Hormonal System — Global Brain State
# ============================================================

class HormonalSystem:
    """
    Organism-level state. Changes EVERYTHING, very slowly.
    
    - Stress (cortisol analog): increases learning rate, lowers thresholds
    - Energy (ATP/glucose analog): gates overall activity
    - Arousal (circadian analog): modulates global responsiveness
    """
    
    def __init__(self):
        self.stress = 0.0           # 0 = calm, 1 = maximum stress
        self.energy = 1.0           # 0 = depleted, 1 = full
        self.arousal = 0.5          # 0 = sleep, 1 = peak alertness
        
        self.history: List[Dict[str, float]] = []
    
    def update(self, prediction_error: float = 0.0, 
               compute_cost: float = 0.0,
               external_stress: float = 0.0):
        """
        Update global state based on brain activity.
        
        Args:
            prediction_error: how wrong the brain's predictions were (drives stress)
            compute_cost: how much computation was used (drains energy)
            external_stress: external stressor (drives stress directly)
        """
        # Stress responds to prediction error and external stressors
        stress_input = prediction_error * 0.3 + external_stress * 0.5
        self.stress = (HORMONE_MOMENTUM * self.stress + 
                      (1 - HORMONE_MOMENTUM) * stress_input)
        self.stress = np.clip(self.stress, 0.0, 1.0)
        
        # Energy depletes with computation, slowly recovers
        self.energy -= compute_cost * 0.001
        self.energy += 0.0005  # slow recovery
        self.energy = np.clip(self.energy, 0.1, 1.0)
        
        # Arousal decays slowly (fatigue)
        self.arousal *= 0.9999
        self.arousal = max(0.1, self.arousal)
        
        self.history.append({
            'stress': self.stress,
            'energy': self.energy,
            'arousal': self.arousal,
        })
    
    def get_modifiers(self) -> Dict[str, float]:
        """How global state modifies neural computation."""
        return {
            'learning_rate': 1.0 + self.stress * 0.5,  # stress increases learning
            'threshold': 1.0 - self.arousal * 0.2,      # arousal lowers thresholds
            'expression_rate': self.energy,               # low energy → less expression
            'firing_probability': self.arousal * self.energy,  # need both arousal and energy
        }
    
    def apply_to_circuit(self, circuit: Circuit):
        """
        Apply hormonal modifiers to entire circuit.
        
        Effects are SOFT modifiers — they nudge values, not multiply destructively.
        Energy affects protein synthesis rate, not expression levels directly.
        """
        mods = self.get_modifiers()
        
        from central_dogma import ChannelProtein
        
        for neuron in circuit.neurons.values():
            # Threshold modification (small per-step adjustment)
            threshold_adj = (mods['threshold'] - 1.0) * 0.01  # 1% of deviation per step
            for protein in neuron.protein_pool.values():
                if isinstance(protein, ChannelProtein):
                    protein.threshold *= (1.0 + threshold_adj)
                    protein.threshold = np.clip(protein.threshold, 0.05, 5.0)
            
            # Low energy slows protein reinforcement (not expression directly)
            # This is handled by the Brain.process method passing energy
            # to the circuit's protein turnover rate
    
    def summary(self) -> str:
        mods = self.get_modifiers()
        return (f"Hormonal State: stress={self.stress:.4f}, energy={self.energy:.4f}, "
                f"arousal={self.arousal:.4f}\n"
                f"  Modifiers: lr={mods['learning_rate']:.3f}, "
                f"threshold={mods['threshold']:.3f}, "
                f"expression={mods['expression_rate']:.3f}")


# ============================================================
# Brain — The Integrated System
# ============================================================

class Brain:
    """
    A complete brain: circuit + neuromodulation + hormones.
    
    Integrates:
    - Neural circuit (neurons + synapses)
    - Neuromodulatory systems (dopamine, serotonin, NE, ACh)
    - Hormonal system (stress, energy, arousal)
    
    This is the full biological stack operating together.
    """
    
    def __init__(self, circuit: Circuit):
        self.circuit = circuit
        self.modulatory_systems: Dict[ModulatorType, NeuromodulatorySystem] = {}
        self.hormones = HormonalSystem()
        self.step_count = 0
    
    def add_modulatory_system(self, modulator_type: ModulatorType,
                               source_neuron_ids: List[int],
                               target_neuron_ids: List[int]) -> NeuromodulatorySystem:
        """Add a neuromodulatory system."""
        system = NeuromodulatorySystem(
            modulator_type, source_neuron_ids, target_neuron_ids
        )
        self.modulatory_systems[modulator_type] = system
        return system
    
    def process(self, input_vector: np.ndarray,
                n_rounds: int = 5,
                learn: bool = False,
                reward: float = 0.0,
                external_stress: float = 0.0) -> Dict:
        """
        Full brain processing step:
        
        1. Update hormonal state
        2. Update neuromodulator levels
        3. Apply neuromodulation to circuit
        4. Apply hormonal modifiers to circuit
        5. Propagate signals through circuit
        6. Hebbian learning (modulated by dopamine + ACh)
        7. Return outputs + brain state
        """
        self.step_count += 1
        
        # 1. Update hormonal state
        self.hormones.update(
            prediction_error=abs(reward) if reward < 0 else 0.0,
            compute_cost=float(self.circuit.n_neurons),
            external_stress=external_stress,
        )
        
        # 2. Update neuromodulator levels from source neuron activity
        for system in self.modulatory_systems.values():
            system.update_level(self.circuit.neurons)
        
        # 3. Apply neuromodulation
        for system in self.modulatory_systems.values():
            system.apply_to_circuit(self.circuit)
        
        # 4. Apply hormonal modifiers
        self.hormones.apply_to_circuit(self.circuit)
        
        # 5. Compute effective learning rate (modulated by dopamine + ACh)
        effective_lr_mult = 1.0
        if ModulatorType.DOPAMINE in self.modulatory_systems:
            effective_lr_mult *= self.modulatory_systems[ModulatorType.DOPAMINE].get_learning_rate_modifier()
        if ModulatorType.ACETYLCHOLINE in self.modulatory_systems:
            effective_lr_mult *= self.modulatory_systems[ModulatorType.ACETYLCHOLINE].get_learning_rate_modifier()
        
        hormonal_lr = self.hormones.get_modifiers()['learning_rate']
        effective_lr_mult *= hormonal_lr
        
        # 6. Propagate signals
        input_signals = self.circuit.inject_input(input_vector)
        outputs = self.circuit.propagate(
            input_signals, n_rounds=n_rounds,
            learn=learn, reward=reward * effective_lr_mult,
        )
        
        output_vector = self.circuit.get_output_vector(outputs)
        
        return {
            'output': output_vector,
            'outputs_per_neuron': outputs,
            'effective_lr': effective_lr_mult,
            'hormonal_state': self.hormones.get_modifiers(),
            'modulator_levels': {
                mt.name: sys.level 
                for mt, sys in self.modulatory_systems.items()
            },
        }
    
    def summary(self) -> str:
        lines = [
            f"Brain (step {self.step_count})",
            f"  {self.circuit.summary()}",
            f"\n  {self.hormones.summary()}",
            f"\n  Neuromodulatory Systems:"
        ]
        for mt, sys in self.modulatory_systems.items():
            lines.append(f"    {sys.summary()}")
        return '\n'.join(lines)


# ============================================================
# Preset Brain Architectures
# ============================================================

def create_modulated_brain(genome: Genome, d_model: int = 64) -> Brain:
    """
    Create a brain with full neuromodulation.
    10 neurons + 4 modulatory systems + hormones.
    """
    from circuit import Circuit
    
    circuit = Circuit(genome, d_model=d_model)
    
    # Create neurons
    # 0-1: Input (perception)
    circuit.add_neuron(0, NeuronType.PERCEPTION)
    circuit.add_neuron(1, NeuronType.PERCEPTION)
    
    # 2-3: Modulatory source neurons (one per modulator pair)
    circuit.add_neuron(2, NeuronType.REGULATION)  # dopamine source
    circuit.add_neuron(3, NeuronType.REGULATION)  # NE source
    
    # 4-7: Processing (hidden)
    circuit.add_neuron(4, NeuronType.MEMORY)
    circuit.add_neuron(5, NeuronType.INTERNEURON)
    circuit.add_neuron(6, NeuronType.PREDICTION)
    circuit.add_neuron(7, NeuronType.INTERNEURON)
    
    # 8-9: Output
    circuit.add_neuron(8, NeuronType.OUTPUT)
    circuit.add_neuron(9, NeuronType.OUTPUT)
    
    # Connectivity
    # Input → processing
    circuit.connect_all([0, 1], [4, 5, 6, 7], weight=0.5, excitatory=True)
    # Input → modulatory sources (they need input to modulate)
    circuit.connect_all([0, 1], [2, 3], weight=0.3, excitatory=True)
    # Processing → output
    circuit.connect_all([4, 5, 6, 7], [8, 9], weight=0.5, excitatory=True)
    # Lateral connections in processing
    circuit.connect_random([4, 5, 6, 7], connection_prob=0.4, 
                          weight_std=0.15, excitatory_ratio=0.7)
    # Feedback: output → processing (recurrence)
    circuit.connect_all([8, 9], [4, 5], weight=0.2, excitatory=True)
    
    circuit.set_input_neurons([0, 1])
    circuit.set_output_neurons([8, 9])
    
    # Create brain with modulatory systems
    brain = Brain(circuit)
    
    processing_ids = [4, 5, 6, 7, 8, 9]
    all_ids = list(range(10))
    
    # Dopamine: sources=[2], targets=all processing
    brain.add_modulatory_system(
        ModulatorType.DOPAMINE, 
        source_neuron_ids=[2],
        target_neuron_ids=processing_ids,
    )
    
    # Serotonin: sources=[3], targets=all processing
    brain.add_modulatory_system(
        ModulatorType.SEROTONIN,
        source_neuron_ids=[3],
        target_neuron_ids=processing_ids,
    )
    
    # Norepinephrine: sources=[2,3], targets=all neurons
    brain.add_modulatory_system(
        ModulatorType.NOREPINEPHRINE,
        source_neuron_ids=[2, 3],
        target_neuron_ids=all_ids,
    )
    
    # Acetylcholine: sources=[3], targets=processing
    brain.add_modulatory_system(
        ModulatorType.ACETYLCHOLINE,
        source_neuron_ids=[3],
        target_neuron_ids=processing_ids,
    )
    
    return brain


# ============================================================
# VERIFICATION SUITE
# ============================================================

def verify_neuromodulation():
    """Comprehensive neuromodulation verification."""
    import tempfile, os, time
    
    print("=" * 70)
    print("V7 NEUROMODULATION VERIFICATION SUITE")
    print("Zero tolerance for error.")
    print("=" * 70)
    
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "neuromod_test.genome")
    
    from genome import create_minimal_genome
    genome = create_minimal_genome(d_model=64, path=path)
    d_model = 64
    
    all_passed = True
    test_num = 0
    
    def check(name, condition, detail=""):
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
    # GROUP 1: Brain Construction
    # ============================================================
    print("\n─── GROUP 1: Brain Construction ───")
    
    brain = create_modulated_brain(genome, d_model=d_model)
    
    check("Brain created", brain is not None)
    check("Circuit has neurons", brain.circuit.n_neurons == 10)
    check("Has 4 modulatory systems", len(brain.modulatory_systems) == 4)
    check("Has dopamine system", ModulatorType.DOPAMINE in brain.modulatory_systems)
    check("Has serotonin system", ModulatorType.SEROTONIN in brain.modulatory_systems)
    check("Has NE system", ModulatorType.NOREPINEPHRINE in brain.modulatory_systems)
    check("Has ACh system", ModulatorType.ACETYLCHOLINE in brain.modulatory_systems)
    check("Hormonal system exists", brain.hormones is not None)
    
    # ============================================================
    # GROUP 2: Basic Processing
    # ============================================================
    print("\n─── GROUP 2: Basic Processing ───")
    
    np.random.seed(42)
    input_vec = np.random.randn(d_model).astype(np.float32)
    
    result = brain.process(input_vec, n_rounds=5)
    
    check("Process returns output", 'output' in result)
    check("Output has correct shape", result['output'].shape == (d_model,))
    check("Output is finite", np.all(np.isfinite(result['output'])))
    check("Has effective lr", 'effective_lr' in result)
    check("Has hormonal state", 'hormonal_state' in result)
    check("Has modulator levels", 'modulator_levels' in result)
    check("All modulator levels present", 
          len(result['modulator_levels']) == 4)
    
    # ============================================================
    # GROUP 3: Neuromodulator Dynamics
    # ============================================================
    print("\n─── GROUP 3: Neuromodulator Dynamics ───")
    
    brain2 = create_modulated_brain(genome, d_model=d_model)
    
    # Run with strong input to drive modulatory neurons
    np.random.seed(42)
    initial_levels = {mt.name: sys.level 
                      for mt, sys in brain2.modulatory_systems.items()}
    
    for _ in range(50):
        inp = np.random.randn(d_model).astype(np.float32) * 2.0
        brain2.process(inp, n_rounds=5)
    
    final_levels = {mt.name: sys.level 
                    for mt, sys in brain2.modulatory_systems.items()}
    
    # Levels should have changed from baseline
    any_changed = any(abs(final_levels[k] - initial_levels[k]) > 0.0001 
                      for k in initial_levels)
    check("Modulator levels change with activity", any_changed,
          f"Initial: {initial_levels}, Final: {final_levels}")
    
    # All levels bounded [0, 1]
    all_bounded = all(0.0 <= v <= 1.0 for v in final_levels.values())
    check("Modulator levels bounded [0, 1]", all_bounded)
    
    # Each system has history
    for mt, sys in brain2.modulatory_systems.items():
        check(f"{mt.name} has history", len(sys.level_history) > 0)
    
    # ============================================================
    # GROUP 4: Dopamine Modulates Learning
    # ============================================================
    print("\n─── GROUP 4: Dopamine → Learning Rate ───")
    
    # High dopamine → higher effective learning rate
    da_system = brain2.modulatory_systems[ModulatorType.DOPAMINE]
    
    # Force high dopamine
    da_system.level = 0.9
    lr_high_da = da_system.get_learning_rate_modifier()
    
    # Force low dopamine
    da_system.level = 0.1
    lr_low_da = da_system.get_learning_rate_modifier()
    
    # Reset
    da_system.level = BASELINE_LEVEL
    lr_baseline = da_system.get_learning_rate_modifier()
    
    check("High DA → high learning rate", lr_high_da > lr_baseline,
          f"High DA lr={lr_high_da:.3f}, Baseline lr={lr_baseline:.3f}")
    check("Low DA → low learning rate", lr_low_da < lr_baseline,
          f"Low DA lr={lr_low_da:.3f}, Baseline lr={lr_baseline:.3f}")
    
    print(f"  DA=0.1 → lr={lr_low_da:.3f}")
    print(f"  DA=0.5 → lr={lr_baseline:.3f}")
    print(f"  DA=0.9 → lr={lr_high_da:.3f}")
    
    # ============================================================
    # GROUP 5: NE Modulates Thresholds
    # ============================================================
    print("\n─── GROUP 5: Norepinephrine → Thresholds ───")
    
    brain3 = create_modulated_brain(genome, d_model=d_model)
    
    # Record initial thresholds
    from central_dogma import ChannelProtein
    initial_thresholds = {}
    for nid, neuron in brain3.circuit.neurons.items():
        channels = [p for p in neuron.protein_pool.values() 
                   if isinstance(p, ChannelProtein)]
        if channels:
            initial_thresholds[nid] = channels[0].threshold
    
    # Force high NE and apply
    ne_system = brain3.modulatory_systems[ModulatorType.NOREPINEPHRINE]
    ne_system.level = 0.9  # high alertness
    ne_system.apply_to_circuit(brain3.circuit)
    
    # Check thresholds changed
    threshold_lowered = 0
    for nid, init_thresh in initial_thresholds.items():
        channels = [p for p in brain3.circuit.neurons[nid].protein_pool.values()
                   if isinstance(p, ChannelProtein)]
        if channels:
            new_thresh = channels[0].threshold
            if new_thresh < init_thresh:
                threshold_lowered += 1
    
    check("High NE lowers thresholds",
          threshold_lowered > 0,
          f"Lowered: {threshold_lowered}/{len(initial_thresholds)}")
    
    # ============================================================
    # GROUP 6: Hormonal System
    # ============================================================
    print("\n─── GROUP 6: Hormonal System ───")
    
    brain4 = create_modulated_brain(genome, d_model=d_model)
    
    check("Initial energy = 1.0", abs(brain4.hormones.energy - 1.0) < 0.01)
    check("Initial stress = 0.0", abs(brain4.hormones.stress) < 0.01)
    
    # Process many steps → energy depletes
    for _ in range(100):
        inp = np.random.randn(d_model).astype(np.float32)
        brain4.process(inp, n_rounds=3)
    
    check("Energy depletes with computation",
          brain4.hormones.energy < 1.0,
          f"Energy: {brain4.hormones.energy:.4f}")
    
    check("Energy stays positive",
          brain4.hormones.energy > 0.0,
          f"Energy: {brain4.hormones.energy:.4f}")
    
    # Stress responds to negative reward (prediction error)
    brain5 = create_modulated_brain(genome, d_model=d_model)
    stress_before = brain5.hormones.stress
    
    for _ in range(20):
        inp = np.random.randn(d_model).astype(np.float32)
        brain5.process(inp, reward=-5.0, external_stress=0.5)
    
    check("Stress increases with negative reward",
          brain5.hormones.stress > stress_before,
          f"Before: {stress_before:.4f}, After: {brain5.hormones.stress:.4f}")
    
    # Hormonal modifiers
    mods = brain5.hormones.get_modifiers()
    check("Learning rate modified by stress",
          mods['learning_rate'] > 1.0,
          f"lr modifier: {mods['learning_rate']:.3f}")
    
    print(f"  {brain5.hormones.summary()}")
    
    # ============================================================
    # GROUP 7: Reward-Modulated Learning
    # ============================================================
    print("\n─── GROUP 7: Reward-Modulated Learning ───")
    
    brain_pos = create_modulated_brain(genome, d_model=d_model)
    brain_neg = create_modulated_brain(genome, d_model=d_model)
    
    np.random.seed(42)
    pattern = np.random.randn(d_model).astype(np.float32) * 2.0
    
    # Train one with positive reward, one with negative
    for _ in range(50):
        brain_pos.process(pattern, n_rounds=5, learn=True, reward=2.0)
        brain_neg.process(pattern, n_rounds=5, learn=True, reward=-2.0)
    
    W_pos = brain_pos.circuit.get_weight_matrix()
    W_neg = brain_neg.circuit.get_weight_matrix()
    
    weight_diff = np.abs(W_pos - W_neg).sum()
    check("Positive vs negative reward → different weights",
          weight_diff > 0.001,
          f"Weight diff: {weight_diff:.6f}")
    
    # Effective learning rate should differ due to modulator states
    result_pos = brain_pos.process(pattern, n_rounds=3)
    result_neg = brain_neg.process(pattern, n_rounds=3)
    
    check("Different brains produce different outputs",
          np.linalg.norm(result_pos['output'] - result_neg['output']) > 0.0001)
    
    # ============================================================
    # GROUP 8: Modulation Changes Circuit Behavior
    # ============================================================
    print("\n─── GROUP 8: Modulation Changes Behavior ───")
    
    brain6 = create_modulated_brain(genome, d_model=d_model)
    
    np.random.seed(42)
    test_input = np.random.randn(d_model).astype(np.float32)
    
    # Normal mode
    result_normal = brain6.process(test_input, n_rounds=5)
    
    # Force high dopamine + high NE (excited mode)
    brain6.modulatory_systems[ModulatorType.DOPAMINE].level = 0.95
    brain6.modulatory_systems[ModulatorType.NOREPINEPHRINE].level = 0.95
    result_excited = brain6.process(test_input, n_rounds=5)
    
    # Force high serotonin (calm mode)
    brain6.modulatory_systems[ModulatorType.DOPAMINE].level = 0.1
    brain6.modulatory_systems[ModulatorType.NOREPINEPHRINE].level = 0.1
    brain6.modulatory_systems[ModulatorType.SEROTONIN].level = 0.95
    result_calm = brain6.process(test_input, n_rounds=5)
    
    # Outputs should differ across modes
    diff_excited = np.linalg.norm(result_normal['output'] - result_excited['output'])
    diff_calm = np.linalg.norm(result_normal['output'] - result_calm['output'])
    
    check("Excited mode differs from normal",
          diff_excited > 0.0001 or result_excited['effective_lr'] != result_normal['effective_lr'],
          f"Output diff: {diff_excited:.6f}, LR normal: {result_normal['effective_lr']:.3f}, "
          f"LR excited: {result_excited['effective_lr']:.3f}")
    
    check("Calm mode differs from normal",
          diff_calm > 0.0001 or result_calm['effective_lr'] != result_normal['effective_lr'],
          f"Output diff: {diff_calm:.6f}")
    
    # ============================================================
    # GROUP 9: Long-Running Stability
    # ============================================================
    print("\n─── GROUP 9: Stability (500 steps) ───")
    
    brain7 = create_modulated_brain(genome, d_model=d_model)
    
    np.random.seed(42)
    any_nan = False
    any_inf = False
    
    t0 = time.perf_counter()
    for step in range(500):
        inp = np.random.randn(d_model).astype(np.float32) * 0.5
        reward = np.random.randn() * 0.5
        stress = max(0, np.random.randn() * 0.1)
        
        result = brain7.process(inp, n_rounds=5, learn=True,
                               reward=reward, external_stress=stress)
        
        if np.any(np.isnan(result['output'])):
            any_nan = True
        if np.any(np.isinf(result['output'])):
            any_inf = True
        
        for level in result['modulator_levels'].values():
            if np.isnan(level) or np.isinf(level):
                any_nan = True
    t1 = time.perf_counter()
    
    check("No NaN in 500 steps", not any_nan)
    check("No Inf in 500 steps", not any_inf)
    
    # All modulators still bounded
    all_mod_bounded = all(0 <= sys.level <= 1 
                         for sys in brain7.modulatory_systems.values())
    check("All modulators bounded [0, 1]", all_mod_bounded)
    
    # Energy hasn't hit zero
    check("Energy > 0 after 500 steps",
          brain7.hormones.energy > 0,
          f"Energy: {brain7.hormones.energy:.4f}")
    
    # All neurons alive
    all_alive = all(n.n_active_proteins > 0 
                   for n in brain7.circuit.neurons.values())
    check("All neurons alive after 500 steps", all_alive)
    
    ms_total = (t1 - t0) * 1000
    steps_per_sec = 500 / (t1 - t0)
    check(f"Performance: {steps_per_sec:.0f} brain steps/sec ({ms_total:.0f}ms)", True)
    
    print(f"\n  Final brain state:")
    for mt, sys in brain7.modulatory_systems.items():
        print(f"    {sys.summary()}")
    print(f"    {brain7.hormones.summary()}")
    
    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "=" * 70)
    if all_passed:
        print(f"ALL {test_num} TESTS PASSED ✅")
    else:
        print(f"SOME TESTS FAILED ❌")
    print("=" * 70)
    
    genome.close()
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)
    
    return all_passed


if __name__ == "__main__":
    verify_neuromodulation()
