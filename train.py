"""
PID-Transformer Training Script

Optimized for Apple M2 MacBook with 8GB RAM.
Trains on Shakespeare text (small, self-contained, no downloads needed).

Usage:
    python train.py                    # train PID-Transformer
    python train.py --baseline         # train Standard Transformer for comparison
    python train.py --compare          # train both and compare

Hardware detection:
    - Apple Silicon → MPS backend
    - NVIDIA GPU → CUDA
    - Fallback → CPU
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import argparse
import time
import json
import os
import urllib.request
from pathlib import Path

from pid_attention import PIDTransformer, StandardTransformer


# ============================================================
# Device Selection
# ============================================================

def get_device():
    if torch.backends.mps.is_available():
        print("🍎 Using Apple Silicon (MPS)")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print(f"🟢 Using CUDA ({torch.cuda.get_device_name(0)})")
        return torch.device("cuda")
    else:
        print("💻 Using CPU")
        return torch.device("cpu")


# ============================================================
# Data: Shakespeare (self-contained)
# ============================================================

SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

class CharDataset(Dataset):
    """Character-level dataset from text file."""
    
    def __init__(self, text: str, seq_len: int = 256):
        self.seq_len = seq_len
        
        # Build vocab
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for ch, i in self.char_to_idx.items()}
        
        # Encode
        self.data = torch.tensor([self.char_to_idx[c] for c in text], dtype=torch.long)
        
    def __len__(self):
        return (len(self.data) - self.seq_len - 1) // self.seq_len
    
    def __getitem__(self, idx):
        start = idx * self.seq_len
        x = self.data[start:start + self.seq_len]
        y = self.data[start + 1:start + self.seq_len + 1]
        return x, y
    
    def decode(self, indices):
        return ''.join(self.idx_to_char[i.item()] for i in indices)


def load_shakespeare(data_dir: str = "data"):
    """Download Shakespeare text if needed."""
    os.makedirs(data_dir, exist_ok=True)
    filepath = os.path.join(data_dir, "shakespeare.txt")
    
    if not os.path.exists(filepath):
        print("📥 Downloading Shakespeare...")
        urllib.request.urlretrieve(SHAKESPEARE_URL, filepath)
    
    with open(filepath, 'r') as f:
        text = f.read()
    
    print(f"📖 Loaded {len(text):,} characters")
    return text


# ============================================================
# Training Loop
# ============================================================

def train_model(
    model: nn.Module,
    train_dataset: CharDataset,
    val_dataset: CharDataset,
    device: torch.device,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 3e-4,
    model_name: str = "model",
    log_interval: int = 50,
    save_dir: str = "checkpoints",
):
    """Train a model and return metrics."""
    
    os.makedirs(save_dir, exist_ok=True)
    
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    
    # Cosine schedule with warmup
    total_steps = len(train_dataset) // batch_size * epochs
    warmup_steps = total_steps // 10
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, drop_last=True)
    
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_ppl": [],
        "epoch_time": [],
        "gate_stats": [],
    }
    
    best_val = float('inf')
    global_step = 0
    
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Device: {device}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {epochs}")
    print(f"  Steps/epoch: {len(train_loader)}")
    print(f"{'='*60}\n")
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        n_batches = 0
        t0 = time.time()
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            out = model(x, targets=y)
            loss = out["loss"]
            loss.backward()
            
            # Gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1
            
            if (batch_idx + 1) % log_interval == 0:
                avg = epoch_loss / n_batches
                lr_now = scheduler.get_last_lr()[0]
                print(f"  [{model_name}] Ep {epoch+1} Step {batch_idx+1}/{len(train_loader)} | "
                      f"Loss {avg:.4f} | PPL {2**avg:.1f} | LR {lr_now:.2e}")
        
        epoch_time = time.time() - t0
        train_loss = epoch_loss / n_batches
        
        # Validation
        model.eval()
        val_loss = 0
        val_batches = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x, targets=y)
                val_loss += out["loss"].item()
                val_batches += 1
        
        val_loss /= max(val_batches, 1)
        val_ppl = 2 ** val_loss  # bits → perplexity
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_ppl"].append(val_ppl)
        history["epoch_time"].append(epoch_time)
        
        # Gate stats for PID model
        if hasattr(model, 'get_gate_stats'):
            history["gate_stats"].append(model.get_gate_stats())
        
        # Save best
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, f"{model_name}_best.pt"))
        
        print(f"  [{model_name}] Epoch {epoch+1}/{epochs} | "
              f"Train {train_loss:.4f} | Val {val_loss:.4f} | "
              f"PPL {val_ppl:.1f} | Time {epoch_time:.1f}s")
    
    history["best_val_loss"] = best_val
    history["best_val_ppl"] = 2 ** best_val
    history["total_params"] = sum(p.numel() for p in model.parameters())
    
    return history


# ============================================================
# Generation Demo
# ============================================================

def demo_generate(model, dataset, device, prompt_text="ROMEO:", max_tokens=200):
    """Generate text from a prompt."""
    model.eval()
    
    # Encode prompt
    indices = [dataset.char_to_idx.get(c, 0) for c in prompt_text]
    input_ids = torch.tensor([indices], dtype=torch.long, device=device)
    
    # Generate
    with torch.no_grad():
        generated = model.generate(input_ids, max_new_tokens=max_tokens, temperature=0.8, top_k=40)
    
    text = dataset.decode(generated[0])
    return text


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Train PID-Transformer")
    parser.add_argument("--baseline", action="store_true", help="Train standard transformer only")
    parser.add_argument("--compare", action="store_true", help="Train both and compare")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--generate", action="store_true", help="Generate text after training")
    args = parser.parse_args()
    
    device = get_device()
    
    # Load data
    text = load_shakespeare()
    split = int(0.9 * len(text))
    train_data = CharDataset(text[:split], seq_len=args.seq_len)
    val_data = CharDataset(text[split:], seq_len=args.seq_len)
    
    print(f"Vocab size: {train_data.vocab_size}")
    print(f"Train sequences: {len(train_data)}")
    print(f"Val sequences: {len(val_data)}")
    
    d_ff = args.d_model * 2  # keep FFN small for 8GB
    
    results = {}
    
    # --- PID-Transformer ---
    if not args.baseline:
        pid_model = PIDTransformer(
            vocab_size=train_data.vocab_size,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            d_ff=d_ff,
            max_seq_len=args.seq_len,
        )
        
        pid_history = train_model(
            pid_model, train_data, val_data, device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            model_name="PID-Transformer",
        )
        results["PID-Transformer"] = pid_history
        
        if args.generate:
            print("\n📝 PID-Transformer Generation:")
            print("-" * 40)
            text_out = demo_generate(pid_model, train_data, device)
            print(text_out)
            print("-" * 40)
    
    # --- Standard Transformer ---
    if args.baseline or args.compare:
        std_model = StandardTransformer(
            vocab_size=train_data.vocab_size,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            d_ff=d_ff,
            max_seq_len=args.seq_len,
        )
        
        std_history = train_model(
            std_model, train_data, val_data, device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            model_name="Std-Transformer",
        )
        results["Std-Transformer"] = std_history
        
        if args.generate:
            print("\n📝 Standard Transformer Generation:")
            print("-" * 40)
            text_out = demo_generate(std_model, train_data, device)
            print(text_out)
            print("-" * 40)
    
    # === Comparison ===
    if len(results) > 1:
        print("\n" + "=" * 60)
        print("COMPARISON")
        print("=" * 60)
        
        print(f"\n{'Model':<20} {'Params':>10} {'Best Val Loss':>14} {'Best PPL':>10}")
        print("-" * 56)
        for name, hist in results.items():
            print(f"{name:<20} {hist['total_params']:>10,} "
                  f"{hist['best_val_loss']:>14.4f} {hist['best_val_ppl']:>10.1f}")
        
        names = list(results.keys())
        if len(names) == 2:
            pid_ppl = results[names[0]]["best_val_ppl"]
            std_ppl = results[names[1]]["best_val_ppl"]
            if pid_ppl < std_ppl:
                print(f"\n🏆 PID-Transformer wins! PPL {pid_ppl:.1f} vs {std_ppl:.1f} "
                      f"({(1-pid_ppl/std_ppl)*100:.1f}% better)")
            else:
                print(f"\n📊 Standard wins this round. PPL {std_ppl:.1f} vs {pid_ppl:.1f}")
    
    # Gate analysis
    if "PID-Transformer" in results and results["PID-Transformer"]["gate_stats"]:
        print("\n🔍 Final Gate Biases (learned P/I/D preferences):")
        final_gates = results["PID-Transformer"]["gate_stats"][-1]
        for layer_name, stats in final_gates.items():
            bias = stats["gate_bias"]
            dominant = ["P", "I", "D"][bias.index(max(bias))]
            print(f"  {layer_name}: P={bias[0]:.3f} I={bias[1]:.3f} D={bias[2]:.3f} → {dominant}-dominant")
    
    # Save results
    Path("results").mkdir(exist_ok=True)
    save_results = {}
    for name, hist in results.items():
        save_results[name] = {
            "total_params": hist["total_params"],
            "best_val_loss": hist["best_val_loss"],
            "best_val_ppl": hist["best_val_ppl"],
            "final_train_loss": hist["train_loss"][-1],
            "epoch_times": hist["epoch_time"],
        }
    
    with open("results/training_results.json", "w") as f:
        json.dump(save_results, f, indent=2)
    
    print(f"\n✅ Results saved to results/training_results.json")
    print(f"✅ Best model saved to checkpoints/")


if __name__ == "__main__":
    main()
