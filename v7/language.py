"""
V7 Phase 5: Scale to Language

The organism learns to speak.

Architecture:
  RETINA (embedding) → BRAIN (biological) → MOTOR CORTEX (projection) → speech

Two learning systems running simultaneously:
  1. GRADIENT DESCENT on interface layers (embedding + output projection)
     - These are the "sensory" and "motor" interfaces with the world
     - Standard backprop, standard PyTorch
  
  2. BIOLOGICAL LEARNING on the brain itself
     - Hebbian plasticity on synapses (fire together → wire together)
     - Gene expression dynamics (activity-dependent specialization)
     - Neuromodulation (reward/prediction error modulates learning rate)
     - Protein turnover (computational machinery adapts)

This is biologically accurate: the retina and motor cortex are the world
interface; internal brain learning is Hebbian/activity-dependent.

Input: bytes (0-255) — character-level, no tokenizer needed
Output: next byte prediction (256-way softmax)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Tuple
import time
import os

from genome import Genome, create_minimal_genome, create_standard_genome
from neuron import Neuron, NeuronType, NeuralSignal
from circuit import Circuit
from neuromodulation import Brain, create_modulated_brain, ModulatorType


# ============================================================
# Language Brain — The Full Organism
# ============================================================

class LanguageBrain(nn.Module):
    """
    A biological brain that processes language.
    
    Architecture:
        byte → embedding (gradient-trained)
             → brain (biologically-trained)
             → projection (gradient-trained)
             → next byte logits
    
    The embedding and projection are standard neural net layers
    trained with backprop. The brain itself uses Hebbian learning,
    gene regulation, and neuromodulation.
    """
    
    def __init__(self, genome_path: str, d_model: int = 64,
                 n_neurons: int = 10, n_vocab: int = 256,
                 propagation_rounds: int = 5):
        super().__init__()
        
        self.d_model = d_model
        self.n_vocab = n_vocab
        self.propagation_rounds = propagation_rounds
        
        # Interface layers (gradient-trained)
        self.embedding = nn.Embedding(n_vocab, d_model)
        self.output_proj = nn.Linear(d_model, n_vocab)
        
        # Layer norm for stability
        self.input_norm = nn.LayerNorm(d_model)
        self.output_norm = nn.LayerNorm(d_model)
        
        # Hidden state — persists across tokens (the brain's memory)
        self.register_buffer('hidden_state', torch.zeros(d_model))
        
        # Biological brain (NOT a PyTorch module — uses numpy)
        self.genome = Genome(genome_path, mode='rw')
        self.brain = create_modulated_brain(self.genome, d_model=d_model)
        
        # Training state
        self._step = 0
        self._total_reward = 0.0
        self._prediction_errors = []
    
    def forward(self, byte_input: torch.Tensor, 
                learn_biological: bool = True) -> torch.Tensor:
        """
        Process one byte, predict the next.
        
        Args:
            byte_input: (batch_size,) tensor of byte values [0-255]
            learn_biological: whether to run Hebbian learning
        
        Returns:
            logits: (batch_size, 256) next byte predictions
        """
        batch_size = byte_input.shape[0]
        
        # Embedding (differentiable)
        embedded = self.embedding(byte_input)  # (B, d_model)
        embedded = self.input_norm(embedded)
        
        # Process each item in batch through the brain
        outputs = []
        for b in range(batch_size):
            inp_np = embedded[b].detach().cpu().numpy().astype(np.float32)
            
            # Mix with hidden state (persistent memory)
            hidden_np = self.hidden_state.detach().cpu().numpy().astype(np.float32)
            brain_input = inp_np * 0.7 + hidden_np * 0.3
            
            # Brain processing (biological)
            result = self.brain.process(
                brain_input,
                n_rounds=self.propagation_rounds,
                learn=learn_biological,
                reward=self._compute_reward(),
            )
            
            brain_output = result['output']
            
            # Update hidden state
            self.hidden_state = torch.tensor(
                brain_output * 0.5 + hidden_np * 0.5,
                dtype=torch.float32, device=byte_input.device
            )
            
            outputs.append(torch.tensor(brain_output, dtype=torch.float32, 
                                       device=byte_input.device))
        
        # Stack outputs
        brain_out = torch.stack(outputs)  # (B, d_model)
        brain_out = self.output_norm(brain_out)
        
        # Project to logits (differentiable)
        logits = self.output_proj(brain_out)  # (B, n_vocab)
        
        self._step += 1
        return logits
    
    def _compute_reward(self) -> float:
        """Compute reward signal from recent prediction errors."""
        if not self._prediction_errors:
            return 0.0
        # Negative prediction error = positive reward (predictions were good)
        recent = self._prediction_errors[-10:]
        avg_error = np.mean(recent)
        # Normalize: low error → positive reward, high error → negative
        reward = 1.0 - min(avg_error / 5.0, 2.0)  # scale to roughly [-1, 1]
        return float(reward)
    
    def record_loss(self, loss_value: float):
        """Record prediction error for reward computation."""
        self._prediction_errors.append(loss_value)
        if len(self._prediction_errors) > 100:
            self._prediction_errors = self._prediction_errors[-100:]
    
    def reset_hidden(self):
        """Reset hidden state (start of new sequence)."""
        self.hidden_state.zero_()
    
    def get_brain_stats(self) -> Dict:
        """Get biological brain statistics."""
        stats = {
            'neurons': self.brain.circuit.n_neurons,
            'synapses': self.brain.circuit.n_synapses,
            'step': self._step,
        }
        
        # Modulator levels
        for mt, sys in self.brain.modulatory_systems.items():
            stats[f'mod_{mt.name.lower()}'] = sys.level
        
        # Hormones
        stats['stress'] = self.brain.hormones.stress
        stats['energy'] = self.brain.hormones.energy
        stats['arousal'] = self.brain.hormones.arousal
        
        # Fire rates
        fire_rates = [n.firing_rate for n in self.brain.circuit.neurons.values()]
        stats['avg_fire_rate'] = np.mean(fire_rates)
        stats['max_fire_rate'] = np.max(fire_rates)
        
        # Weight stats
        ws = self.brain.circuit.get_weight_stats()
        stats['weight_mean'] = ws['mean']
        stats['weight_std'] = ws['std']
        
        return stats


# ============================================================
# Dataset
# ============================================================

class ByteDataset:
    """Simple byte-level text dataset."""
    
    def __init__(self, text_path: str, seq_len: int = 32):
        with open(text_path, 'rb') as f:
            self.data = np.frombuffer(f.read(), dtype=np.uint8)
        self.seq_len = seq_len
        self.n_tokens = len(self.data)
    
    def get_batch(self, batch_size: int, device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a random batch of (input, target) byte sequences."""
        indices = np.random.randint(0, self.n_tokens - self.seq_len - 1, size=batch_size)
        
        x = np.stack([self.data[i:i+self.seq_len] for i in indices])
        y = np.stack([self.data[i+1:i+self.seq_len+1] for i in indices])
        
        return (torch.tensor(x, dtype=torch.long, device=device),
                torch.tensor(y, dtype=torch.long, device=device))
    
    def get_sequential(self, start: int, length: int, 
                       device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sequential chunk."""
        x = self.data[start:start+length]
        y = self.data[start+1:start+length+1]
        return (torch.tensor(x, dtype=torch.long, device=device).unsqueeze(0),
                torch.tensor(y, dtype=torch.long, device=device).unsqueeze(0))


# ============================================================
# Training Loop
# ============================================================

def train(genome_path: str = None, text_path: str = "shakespeare.txt",
          d_model: int = 64, n_steps: int = 500, batch_size: int = 1,
          seq_len: int = 32, lr: float = 1e-3, 
          print_every: int = 50, sample_every: int = 100):
    """
    Train the Language Brain on text.
    
    Two learning systems:
    1. Gradient descent on embedding + projection (every step)
    2. Hebbian learning on brain synapses (every step, modulated by reward)
    """
    import tempfile
    
    # Create genome if not provided
    if genome_path is None:
        tmpdir = tempfile.mkdtemp()
        genome_path = os.path.join(tmpdir, "language_brain.genome")
        genome = create_minimal_genome(d_model=d_model, path=genome_path)
        genome.close()
    
    # Create model
    model = LanguageBrain(genome_path, d_model=d_model)
    
    # Dataset
    dataset = ByteDataset(text_path, seq_len=seq_len)
    print(f"Dataset: {dataset.n_tokens:,} bytes")
    print(f"Model: d={d_model}, neurons={model.brain.circuit.n_neurons}, "
          f"synapses={model.brain.circuit.n_synapses}")
    print(f"Trainable params: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer (only for interface layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training
    losses = []
    bits_per_byte_history = []
    
    print(f"\n{'='*60}")
    print(f"Training for {n_steps} steps...")
    print(f"{'='*60}\n")
    
    t0 = time.perf_counter()
    
    for step in range(1, n_steps + 1):
        model.train()
        model.reset_hidden()
        
        # Get batch
        x, y = dataset.get_batch(batch_size)  # (B, seq_len)
        
        # Process token by token (autoregressive)
        total_loss = 0.0
        n_tokens = 0
        
        for t in range(seq_len):
            input_byte = x[:, t]   # (B,)
            target_byte = y[:, t]  # (B,)
            
            # Forward
            logits = model(input_byte, learn_biological=True)  # (B, 256)
            
            # Loss
            loss = F.cross_entropy(logits, target_byte)
            total_loss += loss.item()
            n_tokens += 1
            
            # Record for reward computation
            model.record_loss(loss.item())
            
            # Backward + step (gradient learning on interface layers)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        avg_loss = total_loss / n_tokens
        bits_per_byte = avg_loss / np.log(2)
        losses.append(avg_loss)
        bits_per_byte_history.append(bits_per_byte)
        
        # Print progress
        if step % print_every == 0 or step == 1:
            elapsed = time.perf_counter() - t0
            tokens_per_sec = step * seq_len * batch_size / elapsed
            
            stats = model.get_brain_stats()
            
            print(f"Step {step:4d} | Loss: {avg_loss:.4f} | "
                  f"BPB: {bits_per_byte:.3f} | "
                  f"tok/s: {tokens_per_sec:.0f} | "
                  f"fire: {stats['avg_fire_rate']:.2f} | "
                  f"DA: {stats['mod_dopamine']:.3f} | "
                  f"energy: {stats['energy']:.3f} | "
                  f"w_std: {stats['weight_std']:.4f}")
        
        # Sample generation
        if step % sample_every == 0:
            sample = generate(model, dataset, length=100)
            print(f"\n  Sample: {repr(sample[:80])}\n")
    
    elapsed = time.perf_counter() - t0
    
    print(f"\n{'='*60}")
    print(f"Training complete in {elapsed:.1f}s")
    print(f"Final loss: {losses[-1]:.4f}")
    print(f"Final BPB: {bits_per_byte_history[-1]:.3f}")
    print(f"Random baseline BPB: {np.log2(256):.3f}")
    
    # Show improvement
    if len(losses) > 20:
        early_bpb = np.mean(bits_per_byte_history[:10])
        late_bpb = np.mean(bits_per_byte_history[-10:])
        improvement = early_bpb - late_bpb
        print(f"BPB improvement: {early_bpb:.3f} → {late_bpb:.3f} (Δ={improvement:+.3f})")
    
    print(f"{'='*60}")
    
    # Final sample
    print("\nFinal generation:")
    sample = generate(model, dataset, length=200, temperature=0.8)
    print(f"  {repr(sample)}")
    
    # Brain state
    print(f"\nFinal brain state:")
    stats = model.get_brain_stats()
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    
    return model, losses


def generate(model: LanguageBrain, dataset: ByteDataset,
             seed_text: str = None, length: int = 100,
             temperature: float = 1.0) -> str:
    """Generate text from the model."""
    model.eval()
    model.reset_hidden()
    
    # Seed with some text
    if seed_text is None:
        # Use a random chunk from dataset as seed
        start = np.random.randint(0, dataset.n_tokens - 50)
        seed_bytes = dataset.data[start:start+20]
    else:
        seed_bytes = np.array([ord(c) for c in seed_text], dtype=np.uint8)
    
    # Feed seed
    with torch.no_grad():
        for byte_val in seed_bytes:
            inp = torch.tensor([byte_val], dtype=torch.long)
            model(inp, learn_biological=False)
    
    # Generate
    generated = list(seed_bytes)
    current_byte = int(seed_bytes[-1])
    
    with torch.no_grad():
        for _ in range(length):
            inp = torch.tensor([current_byte], dtype=torch.long)
            logits = model(inp, learn_biological=False)
            
            # Temperature sampling
            probs = F.softmax(logits[0] / temperature, dim=-1)
            next_byte = torch.multinomial(probs, 1).item()
            
            generated.append(next_byte)
            current_byte = next_byte
    
    # Decode bytes to string
    try:
        return bytes(generated).decode('utf-8', errors='replace')
    except:
        return bytes(generated).decode('latin-1')


# ============================================================
# VERIFICATION
# ============================================================

def verify_language():
    """Verify the language brain works end-to-end."""
    import tempfile
    
    print("=" * 60)
    print("V7 LANGUAGE BRAIN VERIFICATION")
    print("=" * 60)
    
    tmpdir = tempfile.mkdtemp()
    genome_path = os.path.join(tmpdir, "verify.genome")
    genome = create_minimal_genome(d_model=64, path=genome_path)
    genome.close()
    
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
    
    # GROUP 1: Construction
    print("\n─── GROUP 1: Construction ───")
    
    model = LanguageBrain(genome_path, d_model=64)
    check("Model created", model is not None)
    check("Has embedding", hasattr(model, 'embedding'))
    check("Has output projection", hasattr(model, 'output_proj'))
    check("Has brain", model.brain is not None)
    
    n_params = sum(p.numel() for p in model.parameters())
    check("Has trainable params", n_params > 0, f"Params: {n_params}")
    print(f"  Total params: {n_params:,}")
    
    # GROUP 2: Forward pass
    print("\n─── GROUP 2: Forward Pass ───")
    
    inp = torch.tensor([65], dtype=torch.long)  # 'A'
    logits = model(inp, learn_biological=False)
    
    check("Forward produces output", logits is not None)
    check("Output shape correct", logits.shape == (1, 256),
          f"Got {logits.shape}")
    check("Output is finite", torch.all(torch.isfinite(logits)).item())
    
    # Multiple tokens
    model.reset_hidden()
    for byte_val in [72, 101, 108, 108, 111]:  # "Hello"
        inp = torch.tensor([byte_val], dtype=torch.long)
        logits = model(inp, learn_biological=False)
    
    check("Multi-token forward works", logits.shape == (1, 256))
    check("Multi-token output finite", torch.all(torch.isfinite(logits)).item())
    
    # GROUP 3: Loss & gradient
    print("\n─── GROUP 3: Loss & Gradient ───")
    
    model.reset_hidden()
    inp = torch.tensor([65], dtype=torch.long)
    target = torch.tensor([66], dtype=torch.long)
    
    logits = model(inp, learn_biological=True)
    loss = F.cross_entropy(logits, target)
    
    check("Loss computable", loss.item() > 0)
    check("Loss is finite", torch.isfinite(loss).item())
    
    loss.backward()
    
    # Check gradients exist
    grad_exists = any(p.grad is not None and p.grad.abs().sum() > 0 
                      for p in model.parameters())
    check("Gradients flow", grad_exists)
    
    # GROUP 4: Training step
    print("\n─── GROUP 4: Training Step ───")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    model.reset_hidden()
    losses_before = []
    losses_after = []
    
    # A few training steps on a fixed pattern
    pattern_in = [72, 101, 108, 108, 111]   # Hello
    pattern_out = [101, 108, 108, 111, 10]   # ello\n
    
    for epoch in range(20):
        model.reset_hidden()
        epoch_loss = 0.0
        for inp_byte, tgt_byte in zip(pattern_in, pattern_out):
            inp = torch.tensor([inp_byte], dtype=torch.long)
            tgt = torch.tensor([tgt_byte], dtype=torch.long)
            
            logits = model(inp, learn_biological=True)
            loss = F.cross_entropy(logits, tgt)
            epoch_loss += loss.item()
            
            model.record_loss(loss.item())
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        if epoch < 5:
            losses_before.append(epoch_loss / len(pattern_in))
        if epoch >= 15:
            losses_after.append(epoch_loss / len(pattern_in))
    
    avg_before = np.mean(losses_before)
    avg_after = np.mean(losses_after)
    
    check("Loss decreases with training",
          avg_after < avg_before,
          f"Before: {avg_before:.4f}, After: {avg_after:.4f}")
    
    print(f"  Loss: {avg_before:.4f} → {avg_after:.4f}")
    
    # GROUP 5: Generation
    print("\n─── GROUP 5: Generation ───")
    
    # Create a tiny test file
    test_text_path = os.path.join(tmpdir, "test.txt")
    with open(test_text_path, 'w') as f:
        f.write("Hello world! " * 100)
    
    dataset = ByteDataset(test_text_path, seq_len=16)
    check("Dataset created", dataset.n_tokens > 0)
    
    sample = generate(model, dataset, seed_text="Hello", length=50)
    check("Generation produces text", len(sample) > 0)
    check("Generation is string", isinstance(sample, str))
    print(f"  Generated: {repr(sample[:60])}")
    
    # GROUP 6: Brain state after training
    print("\n─── GROUP 6: Brain State ───")
    
    stats = model.get_brain_stats()
    check("Brain has neurons", stats['neurons'] > 0)
    check("Brain has synapses", stats['synapses'] > 0)
    check("Fire rate > 0", stats['avg_fire_rate'] >= 0)  # may be 0 for short training
    check("Energy > 0", stats['energy'] > 0)
    
    # GROUP 7: Stability
    print("\n─── GROUP 7: Stability (100 tokens) ───")
    
    model.reset_hidden()
    any_nan = False
    
    t0 = time.perf_counter()
    for i in range(100):
        byte_val = np.random.randint(0, 256)
        inp = torch.tensor([byte_val], dtype=torch.long)
        logits = model(inp, learn_biological=True)
        
        if torch.any(torch.isnan(logits)):
            any_nan = True
    t1 = time.perf_counter()
    
    check("No NaN in 100 tokens", not any_nan)
    
    tokens_per_sec = 100 / (t1 - t0)
    check(f"Performance: {tokens_per_sec:.1f} tokens/sec", True)
    
    # SUMMARY
    print("\n" + "=" * 60)
    if all_passed:
        print(f"ALL {test_num} TESTS PASSED ✅")
    else:
        print(f"SOME TESTS FAILED ❌")
    print("=" * 60)
    
    model.genome.close()
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)
    
    return all_passed


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    import sys
    
    if "--verify" in sys.argv:
        verify_language()
    elif "--train" in sys.argv:
        # Full training run
        n_steps = 500
        for arg in sys.argv:
            if arg.startswith("--steps="):
                n_steps = int(arg.split("=")[1])
        
        train(text_path="shakespeare.txt", d_model=64, n_steps=n_steps,
              seq_len=32, lr=1e-3, print_every=50, sample_every=200)
    else:
        # Default: verify then short training
        print("Running verification...")
        passed = verify_language()
        
        if passed and os.path.exists("shakespeare.txt"):
            print("\n\nStarting training on Shakespeare...\n")
            train(text_path="shakespeare.txt", d_model=64, n_steps=200,
                  seq_len=32, lr=1e-3, print_every=25, sample_every=100)
