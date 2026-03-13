"""
ionBrain Training Script

Byte-level language model training on Shakespeare.
Compares ionBrain (PID + entropy patching + sparse attention)
vs Standard Byte Transformer baseline.

Optimized for Apple M2 8GB.

Usage:
    python train_ionbrain.py                      # ionBrain only
    python train_ionbrain.py --compare             # ionBrain vs baseline
    python train_ionbrain.py --size tiny            # smaller model
    python train_ionbrain.py --compare --generate   # full comparison + generation
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

from ionbrain import ionBrain, StandardByteTransformer, create_ionbrain


# ============================================================
# Device
# ============================================================

def get_device():
    if torch.backends.mps.is_available():
        print("🍎 Apple Silicon (MPS)")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print(f"🟢 CUDA ({torch.cuda.get_device_name(0)})")
        return torch.device("cuda")
    print("💻 CPU")
    return torch.device("cpu")


# ============================================================
# Data
# ============================================================

SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

class ByteDataset(Dataset):
    """Byte-level dataset — no tokenizer needed."""
    
    def __init__(self, text: str, seq_len: int = 512):
        self.seq_len = seq_len
        self.data = torch.tensor([ord(c) for c in text], dtype=torch.long)
    
    def __len__(self):
        return (len(self.data) - self.seq_len - 1) // (self.seq_len // 2)  # overlapping
    
    def __getitem__(self, idx):
        start = idx * (self.seq_len // 2)
        x = self.data[start:start + self.seq_len]
        y = self.data[start + 1:start + self.seq_len + 1]
        return x, y


def load_data(seq_len=512):
    os.makedirs("data", exist_ok=True)
    path = "data/shakespeare.txt"
    if not os.path.exists(path):
        print("📥 Downloading Shakespeare...")
        urllib.request.urlretrieve(SHAKESPEARE_URL, path)
    
    with open(path) as f:
        text = f.read()
    
    split = int(0.9 * len(text))
    train_ds = ByteDataset(text[:split], seq_len)
    val_ds = ByteDataset(text[split:], seq_len)
    
    print(f"📖 {len(text):,} bytes | Train: {len(train_ds)} seqs | Val: {len(val_ds)} seqs")
    return train_ds, val_ds


# ============================================================
# Training
# ============================================================

def train_model(model, train_ds, val_ds, device, epochs, batch_size, lr, name, log_every=20):
    model = model.to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1, betas=(0.9, 0.95))
    
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, drop_last=True)
    
    total_steps = len(train_dl) * epochs
    warmup = total_steps // 10
    
    def lr_fn(step):
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))
    
    import math
    sched = optim.lr_scheduler.LambdaLR(opt, lr_fn)
    
    params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*50}")
    print(f"Training {name} | {params:,} params | {device}")
    print(f"{'='*50}")
    
    history = {"train_loss": [], "val_loss": [], "val_bpb": [], "epoch_time": []}
    best_val = float('inf')
    step = 0
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        n = 0
        t0 = time.time()
        
        for batch_idx, (x, y) in enumerate(train_dl):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            
            out = model(x, targets=y)
            loss = out["loss"]
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            
            epoch_loss += loss.item()
            n += 1
            step += 1
            
            if (batch_idx + 1) % log_every == 0:
                avg = epoch_loss / n
                bpb = avg / math.log(2)
                lr_now = sched.get_last_lr()[0]
                extra = ""
                if hasattr(model, 'get_gate_stats'):
                    gates = model.get_gate_stats()
                    if gates:
                        first = list(gates.values())[0]
                        extra = f" | G: P={first['P']:.2f} I={first['I']:.2f} D={first['D']:.2f}"
                print(f"  [{name:>12}] Ep {epoch+1} Step {batch_idx+1}/{len(train_dl)} | "
                      f"Loss {avg:.4f} BPB {bpb:.3f} LR {lr_now:.1e}{extra}")
        
        dt = time.time() - t0
        train_loss = epoch_loss / n
        
        # Validation
        model.eval()
        val_loss = 0
        vn = 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                out = model(x, targets=y)
                val_loss += out["loss"].item()
                vn += 1
        
        val_loss /= max(vn, 1)
        val_bpb = val_loss / math.log(2)
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_bpb"].append(val_bpb)
        history["epoch_time"].append(dt)
        
        if val_loss < best_val:
            best_val = val_loss
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), f"checkpoints/{name}_best.pt")
        
        print(f"  [{name:>12}] Epoch {epoch+1}/{epochs} | "
              f"Train {train_loss:.4f} | Val {val_loss:.4f} | BPB {val_bpb:.3f} | {dt:.1f}s")
    
    history["best_val"] = best_val
    history["best_bpb"] = best_val / math.log(2)
    history["params"] = params
    return history


# ============================================================
# Generation
# ============================================================

def demo_generate(model, device, prompt="ROMEO:", max_bytes=300):
    model.eval()
    x = torch.tensor([[ord(c) for c in prompt]], dtype=torch.long, device=device)
    
    with torch.no_grad():
        generated = model.generate(x, max_new=max_bytes, temperature=0.8, top_k=40)
    
    text = ''.join(chr(min(b, 127)) for b in generated[0].tolist())
    return text


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--size", default="small", choices=["tiny", "small", "base"])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--generate", action="store_true")
    args = parser.parse_args()
    
    device = get_device()
    train_ds, val_ds = load_data(args.seq_len)
    
    results = {}
    
    # ionBrain
    model = create_ionbrain(args.size)
    history = train_model(model, train_ds, val_ds, device, args.epochs, args.batch_size, args.lr, f"ionBrain-{args.size}")
    results[f"ionBrain-{args.size}"] = history
    
    if args.generate and hasattr(model, 'generate'):
        print(f"\n📝 ionBrain Generation:")
        print("-" * 40)
        print(demo_generate(model, device))
        print("-" * 40)
    
    # Gate analysis
    print(f"\n🔍 Final Gate Analysis:")
    model.eval()
    with torch.no_grad():
        x = torch.tensor([[ord(c) for c in "To be, or not to be"]], device=device)
        _ = model(x)
    gates = model.get_gate_stats()
    for name, g in gates.items():
        dominant = max(g, key=g.get)
        print(f"  {name}: P={g['P']:.3f} I={g['I']:.3f} D={g['D']:.3f} → {dominant}")
    
    # Baseline
    if args.compare:
        d_model = {"tiny": 192, "small": 256, "base": 384}[args.size]
        n_layers = {"tiny": 4, "small": 6, "base": 8}[args.size]
        d_ff = d_model * 2
        
        baseline = StandardByteTransformer(
            d_model=d_model, n_heads=4, n_layers=n_layers,
            d_ff=d_ff, max_seq_len=args.seq_len,
        )
        history2 = train_model(baseline, train_ds, val_ds, device, args.epochs, args.batch_size, args.lr, "Std-ByteTF")
        results["Std-ByteTF"] = history2
        
        if args.generate:
            print(f"\n📝 Standard Transformer Generation:")
            print("-" * 40)
            # Standard model doesn't have generate, skip
            print("(generation not implemented for baseline)")
            print("-" * 40)
    
    # Summary
    if len(results) > 1:
        print(f"\n{'='*60}")
        print("COMPARISON")
        print(f"{'='*60}")
        print(f"{'Model':<20} {'Params':>10} {'Best BPB':>10} {'Best Loss':>12}")
        print("-" * 54)
        for name, h in results.items():
            print(f"{name:<20} {h['params']:>10,} {h['best_bpb']:>10.3f} {h['best_val']:>12.4f}")
        
        names = list(results.keys())
        if len(names) == 2:
            bpb1 = results[names[0]]["best_bpb"]
            bpb2 = results[names[1]]["best_bpb"]
            if bpb1 < bpb2:
                print(f"\n🏆 {names[0]} wins! BPB {bpb1:.3f} vs {bpb2:.3f} ({(1-bpb1/bpb2)*100:.1f}% better)")
            else:
                print(f"\n📊 {names[1]} wins this round. BPB {bpb2:.3f} vs {bpb1:.3f}")
    
    # Save
    Path("results").mkdir(exist_ok=True)
    with open("results/ionbrain_results.json", "w") as f:
        json.dump({n: {k: v for k, v in h.items() if k != "gate_stats"} for n, h in results.items()}, f, indent=2, default=str)
    print(f"\n✅ Results saved to results/ionbrain_results.json")


if __name__ == "__main__":
    main()
