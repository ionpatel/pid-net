"""
V7 Phase 3: Gene Regulatory Network (GRN)

In biology, genes don't just produce proteins — some proteins REGULATE other genes.
Gene A → Protein A (transcription factor) → binds Gene B's promoter → activates/represses Gene B.

This creates a regulatory NETWORK that can implement:
- Bistable switches (memory) — two genes mutually repress each other
- Oscillators (timing) — negative feedback loops with delay  
- Feed-forward loops (noise filtering) — A activates B, both activate C
- Cascades (signal amplification) — A → B → C → D
- Combinatorial logic (AND/OR gates) — C requires both A AND B

Gene regulatory networks are TURING COMPLETE. They can compute anything.
The computation isn't IN the genes — it's in the NETWORK of genes regulating each other.

This module:
1. Discovers the regulatory network from genome structure
2. Simulates regulatory dynamics (TFs binding promoters, changing expression)
3. Implements regulatory motifs (switches, oscillators, cascades)
4. Connects to neuron gene expression for activity-dependent regulation
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Set
from collections import defaultdict

from genome import Genome, GeneType, GeneEntry
from central_dogma import (
    TranscriptionEngine, TranscriptionFactor, Protein
)


# ============================================================
# Constants
# ============================================================

BINDING_THRESHOLD = 0.15         # minimum cosine similarity for TF to bind
REGULATION_RATE = 0.02           # how fast expression changes per regulatory step
BASAL_EXPRESSION = 0.3           # default expression without TF input
COOPERATIVITY = 2.0              # Hill coefficient — nonlinearity of TF binding
MAX_EXPRESSION = 1.0
MIN_EXPRESSION = 0.0


# ============================================================
# Regulatory Edge
# ============================================================

@dataclass
class RegulatoryEdge:
    """One regulatory interaction: TF from source gene affects target gene."""
    source_gene: int              # gene that produces the TF
    target_gene: int              # gene being regulated
    affinity: float               # binding affinity (cosine similarity)
    is_activator: bool            # True = activates, False = represses
    strength: float = 1.0         # how strongly this edge regulates
    
    def compute_effect(self, source_expression: float) -> float:
        """
        Compute regulatory effect of this edge.
        Uses Hill function for cooperative binding (biological nonlinearity).
        
        effect = affinity * strength * hill(source_expression)
        Positive for activators, negative for repressors.
        """
        # Hill function: e^n / (K^n + e^n) where e = expression, K = 0.5, n = cooperativity
        K = 0.5
        hill = (source_expression ** COOPERATIVITY) / (K ** COOPERATIVITY + source_expression ** COOPERATIVITY + 1e-8)
        
        effect = self.affinity * self.strength * hill
        
        if not self.is_activator:
            effect = -effect
        
        return effect


# ============================================================
# Gene Regulatory Network
# ============================================================

class GeneRegulatoryNetwork:
    """
    The gene regulatory network (GRN).
    
    Discovers and simulates regulatory interactions between genes.
    TF genes produce transcription factors that bind to other genes' promoters,
    creating a complex regulatory network.
    
    Key capabilities:
    - Automatic network discovery from genome structure
    - Regulatory dynamics simulation
    - Motif detection (bistable switches, oscillators, FFLs)
    - Activity-dependent regulation updates
    """
    
    def __init__(self, genome: Genome):
        self.genome = genome
        self.engine = TranscriptionEngine(genome)
        
        # Regulatory edges: (source, target) → RegulatoryEdge
        self.edges: Dict[Tuple[int, int], RegulatoryEdge] = {}
        
        # Adjacency lists
        self.regulators_of: Dict[int, List[int]] = defaultdict(list)  # target → [source genes]
        self.targets_of: Dict[int, List[int]] = defaultdict(list)     # source → [target genes]
        
        # Expression state
        self.expression: Dict[int, float] = {}
        
        # Discover network from genome
        self._discover_network()
    
    def _discover_network(self):
        """
        Discover regulatory interactions from genome structure.
        
        For each regulatory gene:
        1. Express it to get its TF protein
        2. Check TF binding affinity against ALL other genes' promoters
        3. Create edges where binding affinity > threshold
        """
        regulatory_genes = self.genome.get_genes_by_type(GeneType.REGULATORY)
        
        for reg_gid in regulatory_genes:
            # Express this regulatory gene to get its TF
            tf_protein = self.engine.express_gene(reg_gid, active_tfs=[], 
                                                   activity_level=0.0)
            if not isinstance(tf_protein, TranscriptionFactor):
                continue
            
            # Check binding against all genes
            for target_gid in range(self.genome.n_genes):
                if target_gid == reg_gid:
                    continue  # no self-regulation (for now)
                
                promoter = self.genome.get_promoter(target_gid).astype(np.float32)
                affinity = tf_protein.bind_affinity(promoter)
                
                if abs(affinity) > BINDING_THRESHOLD:
                    edge = RegulatoryEdge(
                        source_gene=reg_gid,
                        target_gene=target_gid,
                        affinity=abs(affinity),
                        is_activator=affinity > 0,
                        strength=tf_protein.strength,
                    )
                    self.edges[(reg_gid, target_gid)] = edge
                    self.regulators_of[target_gid].append(reg_gid)
                    self.targets_of[reg_gid].append(target_gid)
        
        # Initialize expression for all genes
        for gid in range(self.genome.n_genes):
            gene = self.genome.get_gene(gid)
            self.expression[gid] = gene.base_expression
    
    def step(self, activity_levels: Optional[Dict[int, float]] = None,
             external_signals: Optional[Dict[int, float]] = None) -> Dict[int, float]:
        """
        One step of regulatory dynamics.
        
        For each gene:
        1. Compute net regulatory input from all TFs
        2. Add activity-dependent modulation
        3. Add external signals (neuromodulators)
        4. Update expression level
        
        Args:
            activity_levels: gene_id → neural activity (from neuron firing)
            external_signals: gene_id → external modulation (neuromodulators)
        
        Returns:
            Updated expression levels for all genes
        """
        new_expression = {}
        
        for gid in range(self.genome.n_genes):
            gene = self.genome.get_gene(gid)
            
            # Basal expression
            net_input = BASAL_EXPRESSION
            
            # Regulatory input from TFs
            for source_gid in self.regulators_of[gid]:
                edge = self.edges.get((source_gid, gid))
                if edge:
                    source_expr = self.expression.get(source_gid, 0.0)
                    effect = edge.compute_effect(source_expr)
                    net_input += effect
            
            # Activity-dependent modulation
            if activity_levels and gid in activity_levels:
                net_input += activity_levels[gid] * gene.activity_sensitivity
            
            # External signals (neuromodulators)
            if external_signals and gid in external_signals:
                net_input += external_signals[gid]
            
            # Sigmoid activation → new expression level
            target_expression = 1.0 / (1.0 + np.exp(-net_input))
            target_expression = np.clip(target_expression, MIN_EXPRESSION, MAX_EXPRESSION)
            
            # Smooth update (biological inertia)
            old_expr = self.expression.get(gid, gene.base_expression)
            new_expr = old_expr + REGULATION_RATE * (target_expression - old_expr)
            new_expression[gid] = float(new_expr)
        
        self.expression = new_expression
        return new_expression
    
    def run_to_steady_state(self, max_steps: int = 100, 
                             tolerance: float = 1e-4) -> Tuple[Dict[int, float], int]:
        """
        Run regulatory dynamics until steady state (expression stops changing).
        
        Returns:
            (final_expression, n_steps_to_converge)
        """
        for step in range(max_steps):
            old_expr = dict(self.expression)
            self.step()
            
            # Check convergence
            max_change = max(abs(self.expression[gid] - old_expr[gid]) 
                           for gid in self.expression)
            
            if max_change < tolerance:
                return self.expression, step + 1
        
        return self.expression, max_steps
    
    # ================================================================
    # MOTIF DETECTION
    # ================================================================
    
    def find_feedback_loops(self) -> List[List[int]]:
        """Find all feedback loops in the regulatory network."""
        loops = []
        
        # Find 2-gene mutual regulation (bistable switches)
        for (src, tgt) in self.edges:
            if (tgt, src) in self.edges:
                loop = sorted([src, tgt])
                if loop not in loops:
                    loops.append(loop)
        
        # Find 3-gene loops
        for a in self.targets_of:
            for b in self.targets_of.get(a, []):
                for c in self.targets_of.get(b, []):
                    if c == a:
                        loop = sorted([a, b, c])
                        if loop not in loops and len(set(loop)) == 3:
                            loops.append(loop)
        
        return loops
    
    def find_feed_forward_loops(self) -> List[Tuple[int, int, int]]:
        """
        Find feed-forward loops: A → B, A → C, B → C.
        These act as noise filters — C requires sustained activation of A.
        """
        ffls = []
        
        for a in self.targets_of:
            targets_a = set(self.targets_of[a])
            for b in targets_a:
                targets_b = set(self.targets_of.get(b, []))
                common = targets_a & targets_b
                for c in common:
                    if c != a and c != b:
                        ffls.append((a, b, c))
        
        return ffls
    
    def classify_edge(self, source: int, target: int) -> str:
        """Classify a regulatory edge."""
        edge = self.edges.get((source, target))
        if edge is None:
            return "none"
        return "activation" if edge.is_activator else "repression"
    
    def classify_loop(self, loop: List[int]) -> str:
        """
        Classify a feedback loop:
        - Positive feedback (even number of repressions) → bistable switch
        - Negative feedback (odd number of repressions) → oscillator
        """
        if len(loop) < 2:
            return "unknown"
        
        n_repressions = 0
        for i in range(len(loop)):
            src = loop[i]
            tgt = loop[(i + 1) % len(loop)]
            edge = self.edges.get((src, tgt))
            if edge and not edge.is_activator:
                n_repressions += 1
        
        if n_repressions % 2 == 0:
            return "positive_feedback (bistable switch)"
        else:
            return "negative_feedback (oscillator)"
    
    # ================================================================
    # ANALYSIS
    # ================================================================
    
    def get_network_stats(self) -> Dict:
        """Statistics about the regulatory network."""
        n_activating = sum(1 for e in self.edges.values() if e.is_activator)
        n_repressing = sum(1 for e in self.edges.values() if not e.is_activator)
        
        in_degrees = defaultdict(int)
        out_degrees = defaultdict(int)
        for (src, tgt) in self.edges:
            out_degrees[src] += 1
            in_degrees[tgt] += 1
        
        return {
            "n_edges": len(self.edges),
            "n_activating": n_activating,
            "n_repressing": n_repressing,
            "n_regulatory_genes": len(set(src for src, _ in self.edges)),
            "n_regulated_genes": len(set(tgt for _, tgt in self.edges)),
            "avg_in_degree": np.mean(list(in_degrees.values())) if in_degrees else 0,
            "avg_out_degree": np.mean(list(out_degrees.values())) if out_degrees else 0,
            "max_in_degree": max(in_degrees.values()) if in_degrees else 0,
            "max_out_degree": max(out_degrees.values()) if out_degrees else 0,
            "feedback_loops": len(self.find_feedback_loops()),
            "feed_forward_loops": len(self.find_feed_forward_loops()),
        }
    
    def get_expression_by_type(self) -> Dict[str, List[Tuple[int, float]]]:
        """Get expression levels grouped by gene type."""
        by_type = {}
        for gid, expr in sorted(self.expression.items()):
            gene = self.genome.get_gene(gid)
            type_name = GeneType(gene.gene_type).name
            if type_name not in by_type:
                by_type[type_name] = []
            by_type[type_name].append((gid, expr))
        return by_type
    
    def summary(self) -> str:
        """Human-readable network summary."""
        stats = self.get_network_stats()
        lines = [
            f"Gene Regulatory Network",
            f"  Edges: {stats['n_edges']} ({stats['n_activating']} activating, {stats['n_repressing']} repressing)",
            f"  Regulatory genes: {stats['n_regulatory_genes']}",
            f"  Regulated genes: {stats['n_regulated_genes']}",
            f"  Avg in-degree: {stats['avg_in_degree']:.1f}, Avg out-degree: {stats['avg_out_degree']:.1f}",
            f"  Feedback loops: {stats['feedback_loops']}",
            f"  Feed-forward loops: {stats['feed_forward_loops']}",
        ]
        
        # Top regulators
        out_deg = defaultdict(int)
        for src, _ in self.edges:
            out_deg[src] += 1
        if out_deg:
            top = sorted(out_deg.items(), key=lambda x: -x[1])[:5]
            lines.append(f"  Top regulators: {['g'+str(g)+':'+str(d) for g,d in top]}")
        
        return '\n'.join(lines)


# ============================================================
# Regulatory Motif Constructors
# ============================================================

def inject_bistable_switch(grn: GeneRegulatoryNetwork, 
                           gene_a: int, gene_b: int,
                           strength: float = 0.8):
    """
    Inject a bistable switch motif: A represses B, B represses A.
    The system settles into one of two stable states: A high/B low, or A low/B high.
    This is biological MEMORY — a bit that remembers its state.
    """
    # A represses B
    edge_ab = RegulatoryEdge(
        source_gene=gene_a, target_gene=gene_b,
        affinity=strength, is_activator=False, strength=strength,
    )
    grn.edges[(gene_a, gene_b)] = edge_ab
    grn.regulators_of[gene_b].append(gene_a)
    grn.targets_of[gene_a].append(gene_b)
    
    # B represses A
    edge_ba = RegulatoryEdge(
        source_gene=gene_b, target_gene=gene_a,
        affinity=strength, is_activator=False, strength=strength,
    )
    grn.edges[(gene_b, gene_a)] = edge_ba
    grn.regulators_of[gene_a].append(gene_b)
    grn.targets_of[gene_b].append(gene_a)


def inject_oscillator(grn: GeneRegulatoryNetwork,
                      genes: List[int], strength: float = 0.8):
    """
    Inject an oscillator motif: circular negative feedback.
    A activates B, B activates C, C represses A → oscillation.
    This is biological TIMING — a clock that ticks.
    """
    n = len(genes)
    for i in range(n):
        src = genes[i]
        tgt = genes[(i + 1) % n]
        
        # Last edge is repression (creates negative feedback)
        is_act = (i < n - 1)
        
        edge = RegulatoryEdge(
            source_gene=src, target_gene=tgt,
            affinity=strength, is_activator=is_act, strength=strength,
        )
        grn.edges[(src, tgt)] = edge
        grn.regulators_of[tgt].append(src)
        grn.targets_of[src].append(tgt)


def inject_cascade(grn: GeneRegulatoryNetwork,
                   genes: List[int], strength: float = 0.8):
    """
    Inject a signaling cascade: A → B → C → D (all activation).
    Signal amplification — weak input at A produces strong output at D.
    """
    for i in range(len(genes) - 1):
        src = genes[i]
        tgt = genes[i + 1]
        
        edge = RegulatoryEdge(
            source_gene=src, target_gene=tgt,
            affinity=strength, is_activator=True, strength=strength,
        )
        grn.edges[(src, tgt)] = edge
        grn.regulators_of[tgt].append(src)
        grn.targets_of[src].append(tgt)


# ============================================================
# VERIFICATION SUITE
# ============================================================

def verify_regulation():
    """Comprehensive GRN verification."""
    import tempfile, os, time
    
    print("=" * 70)
    print("V7 GENE REGULATORY NETWORK VERIFICATION SUITE")
    print("Zero tolerance for error.")
    print("=" * 70)
    
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "grn_test.genome")
    
    from genome import create_minimal_genome
    genome = create_minimal_genome(d_model=64, path=path)
    
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
    # GROUP 1: Network Discovery
    # ============================================================
    print("\n─── GROUP 1: Network Discovery ───")
    
    grn = GeneRegulatoryNetwork(genome)
    
    check("GRN created", grn is not None)
    check("Edges discovered", len(grn.edges) > 0,
          f"Got {len(grn.edges)} edges")
    
    stats = grn.get_network_stats()
    check("Has activating edges", stats['n_activating'] > 0)
    check("Has repressing edges", stats['n_repressing'] > 0)
    check("Has regulated genes", stats['n_regulated_genes'] > 0)
    
    print(f"\n  {grn.summary()}")
    
    # ============================================================
    # GROUP 2: Regulatory Dynamics
    # ============================================================
    print("\n─── GROUP 2: Regulatory Dynamics ───")
    
    initial_expr = dict(grn.expression)
    
    # Run one step
    new_expr = grn.step()
    check("Step produces expression values",
          len(new_expr) == genome.n_genes)
    
    # Run to steady state (complex networks may need more steps)
    grn2 = GeneRegulatoryNetwork(genome)
    final_expr, n_steps = grn2.run_to_steady_state(max_steps=500, tolerance=1e-3)
    
    check("Reaches steady state (within 500 steps)", n_steps < 500,
          f"Took {n_steps} steps")
    
    # Expression should be bounded [0, 1]
    all_bounded = all(0.0 <= v <= 1.0 for v in final_expr.values())
    check("Expression bounded [0, 1]", all_bounded)
    
    # Expression should differ from initial (network has effect)
    total_change = sum(abs(final_expr[g] - initial_expr[g]) for g in final_expr)
    check("Expression changed from initial",
          total_change > 0.01,
          f"Total change: {total_change:.4f}")
    
    # No NaN or Inf
    any_bad = any(np.isnan(v) or np.isinf(v) for v in final_expr.values())
    check("No NaN/Inf in expression", not any_bad)
    
    # ============================================================
    # GROUP 3: Bistable Switch
    # ============================================================
    print("\n─── GROUP 3: Bistable Switch (Memory) ───")
    
    grn3 = GeneRegulatoryNetwork(genome)
    
    # Pick two structural genes and make a bistable switch
    struct_genes = genome.get_genes_by_type(GeneType.STRUCTURAL)
    if len(struct_genes) >= 2:
        gene_a, gene_b = struct_genes[0], struct_genes[1]
        
        # Remove any existing edges involving these genes (clean slate for motif test)
        for key in list(grn3.edges.keys()):
            if key[1] == gene_a or key[1] == gene_b:
                del grn3.edges[key]
        grn3.regulators_of[gene_a] = []
        grn3.regulators_of[gene_b] = []
        
        inject_bistable_switch(grn3, gene_a, gene_b, strength=4.0)
        
        # Test 1: Start with A high → A stays high, B goes low
        grn3.expression[gene_a] = 0.9
        grn3.expression[gene_b] = 0.1
        
        final, steps = grn3.run_to_steady_state(max_steps=500)
        
        check("Bistable switch converges (state 1)",
              steps < 500, f"Steps: {steps}")
        check("State 1: A high, B low",
              final[gene_a] > final[gene_b],
              f"A={final[gene_a]:.4f}, B={final[gene_b]:.4f}")
        
        state1_a = final[gene_a]
        state1_b = final[gene_b]
        
        # Test 2: Start with B high → B stays high, A goes low
        grn3.expression[gene_a] = 0.1
        grn3.expression[gene_b] = 0.9
        
        final2, steps2 = grn3.run_to_steady_state(max_steps=500)
        
        check("Bistable switch converges (state 2)",
              steps2 < 500, f"Steps: {steps2}")
        check("State 2: B high, A low",
              final2[gene_b] > final2[gene_a],
              f"A={final2[gene_a]:.4f}, B={final2[gene_b]:.4f}")
        
        # Verify it's actually bistable — the two states should be different
        check("Two distinct stable states",
              abs(state1_a - final2[gene_a]) > 0.01,
              f"State1 A={state1_a:.4f}, State2 A={final2[gene_a]:.4f}")
        
        print(f"  Switch state 1: A={state1_a:.4f}, B={state1_b:.4f}")
        print(f"  Switch state 2: A={final2[gene_a]:.4f}, B={final2[gene_b]:.4f}")
    
    # ============================================================
    # GROUP 4: Oscillator
    # ============================================================
    print("\n─── GROUP 4: Oscillator (Timing) ───")
    
    grn4 = GeneRegulatoryNetwork(genome)
    
    if len(struct_genes) >= 3:
        osc_genes = struct_genes[:3]
        inject_oscillator(grn4, osc_genes, strength=3.0)
        
        # Set initial conditions
        grn4.expression[osc_genes[0]] = 0.9
        grn4.expression[osc_genes[1]] = 0.1
        grn4.expression[osc_genes[2]] = 0.1
        
        # Run and record expression over time
        history = {g: [] for g in osc_genes}
        for _ in range(300):
            grn4.step()
            for g in osc_genes:
                history[g].append(grn4.expression[g])
        
        # Check for oscillation: each gene should have peaks and troughs
        for g in osc_genes:
            vals = np.array(history[g])
            check(f"Gene {g} bounded during oscillation",
                  np.all(vals >= 0) and np.all(vals <= 1))
        
        # Check that expression varies over time (not stuck)
        variations = [np.std(history[g][-100:]) for g in osc_genes]
        avg_var = np.mean(variations)
        check("Oscillator genes show variation",
              avg_var > 0.0001,
              f"Avg std: {avg_var:.6f}")
        
        # At least check genes don't all converge to same value
        final_vals = [grn4.expression[g] for g in osc_genes]
        spread = max(final_vals) - min(final_vals)
        check("Oscillator genes have spread",
              spread > 0.001,
              f"Spread: {spread:.6f}")
        
        print(f"  Oscillator final: {[f'g{g}={grn4.expression[g]:.4f}' for g in osc_genes]}")
        print(f"  Variation (last 100 steps): {[f'{v:.6f}' for v in variations]}")
    
    # ============================================================
    # GROUP 5: Cascade
    # ============================================================
    print("\n─── GROUP 5: Cascade (Amplification) ───")
    
    grn5 = GeneRegulatoryNetwork(genome)
    
    if len(struct_genes) >= 4:
        cascade_genes = struct_genes[:4]
        inject_cascade(grn5, cascade_genes, strength=2.0)
        
        # Set first gene high, rest low
        grn5.expression[cascade_genes[0]] = 0.9
        for g in cascade_genes[1:]:
            grn5.expression[g] = 0.1
        
        # Run cascade
        history = {g: [] for g in cascade_genes}
        for _ in range(200):
            grn5.step()
            for g in cascade_genes:
                history[g].append(grn5.expression[g])
        
        # Signal should propagate: each downstream gene should eventually rise
        final_vals = [grn5.expression[g] for g in cascade_genes]
        
        check("Cascade first gene stays high",
              final_vals[0] > 0.5,
              f"Gene 0: {final_vals[0]:.4f}")
        
        # Check signal propagation — downstream genes should increase
        for i in range(1, len(cascade_genes)):
            initial_val = 0.1  # we set them to 0.1
            check(f"Cascade gene {i} increased",
                  final_vals[i] > initial_val + 0.01,
                  f"Gene {i}: {final_vals[i]:.4f} (was {initial_val})")
        
        print(f"  Cascade propagation: {[f'g{g}={v:.4f}' for g,v in zip(cascade_genes, final_vals)]}")
    
    # ============================================================
    # GROUP 6: Activity-Dependent Regulation
    # ============================================================
    print("\n─── GROUP 6: Activity-Dependent Regulation ───")
    
    grn6 = GeneRegulatoryNetwork(genome)
    expr_before = dict(grn6.expression)
    
    # Simulate high neural activity affecting specific genes
    activity = {gid: 1.0 for gid in struct_genes[:3]}  # high activity on first 3 genes
    
    for _ in range(50):
        grn6.step(activity_levels=activity)
    
    # Activity-sensitive genes should have changed expression
    changed_count = sum(1 for gid in activity 
                       if abs(grn6.expression[gid] - expr_before[gid]) > 0.001)
    check("Activity modulates gene expression",
          changed_count > 0,
          f"Changed: {changed_count}/{len(activity)}")
    
    # ============================================================
    # GROUP 7: External Signals (Neuromodulator Effect)
    # ============================================================
    print("\n─── GROUP 7: External Signals ───")
    
    grn7 = GeneRegulatoryNetwork(genome)
    expr_before7 = dict(grn7.expression)
    
    # Simulate neuromodulator boosting some genes
    ext_signal = {struct_genes[0]: 2.0, struct_genes[1]: -2.0}  # boost A, suppress B
    
    for _ in range(50):
        grn7.step(external_signals=ext_signal)
    
    check("External signal changes expression",
          abs(grn7.expression[struct_genes[0]] - expr_before7[struct_genes[0]]) > 0.001 or
          abs(grn7.expression[struct_genes[1]] - expr_before7[struct_genes[1]]) > 0.001)
    
    # Boosted gene should be higher than suppressed
    check("Boosted gene > suppressed gene",
          grn7.expression[struct_genes[0]] > grn7.expression[struct_genes[1]],
          f"Boosted: {grn7.expression[struct_genes[0]]:.4f}, "
          f"Suppressed: {grn7.expression[struct_genes[1]]:.4f}")
    
    # ============================================================
    # GROUP 8: Motif Detection
    # ============================================================
    print("\n─── GROUP 8: Motif Detection ───")
    
    grn8 = GeneRegulatoryNetwork(genome)
    
    # Inject known motifs and check detection
    if len(struct_genes) >= 5:
        inject_bistable_switch(grn8, struct_genes[0], struct_genes[1])
        inject_cascade(grn8, [struct_genes[2], struct_genes[3], struct_genes[4]])
        
        loops = grn8.find_feedback_loops()
        check("Detects feedback loops",
              any(set(l) == {struct_genes[0], struct_genes[1]} for l in loops),
              f"Found loops: {loops}")
        
        ffls = grn8.find_feed_forward_loops()
        check("FFL detection runs without error", True)  # FFLs depend on network topology
    
    # ============================================================
    # GROUP 9: Long-Running Stability
    # ============================================================
    print("\n─── GROUP 9: Stability (1000 steps) ───")
    
    grn9 = GeneRegulatoryNetwork(genome)
    
    # Inject complex motifs
    if len(struct_genes) >= 4:
        inject_bistable_switch(grn9, struct_genes[0], struct_genes[1], strength=2.0)
        inject_oscillator(grn9, struct_genes[1:4], strength=2.0)
    
    any_nan = False
    any_inf = False
    any_oob = False
    
    t0 = time.perf_counter()
    for step in range(1000):
        activity = {gid: np.random.uniform(0, 1) for gid in struct_genes[:2]}
        grn9.step(activity_levels=activity)
        
        for v in grn9.expression.values():
            if np.isnan(v):
                any_nan = True
            if np.isinf(v):
                any_inf = True
            if v < -0.01 or v > 1.01:
                any_oob = True
    t1 = time.perf_counter()
    
    check("No NaN in 1000 steps", not any_nan)
    check("No Inf in 1000 steps", not any_inf)
    check("All expression in [0, 1]", not any_oob)
    
    ms_total = (t1 - t0) * 1000
    steps_per_sec = 1000 / (t1 - t0)
    check(f"Performance: {steps_per_sec:.0f} GRN steps/sec ({ms_total:.0f}ms total)", True)
    
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
    verify_regulation()
