"""
ionBrain Training v2 — Bigger Datasets + Fair Comparison

Datasets:
  - shakespeare: 1MB char-level (quick test)
  - tinystories: ~500MB children's stories (HuggingFace)
  - wikitext: ~100MB Wikipedia articles (HuggingFace)

Fair comparison: both models get identical param budgets.

Usage:
    # Quick test
    python train_v2.py --dataset shakespeare --size tiny --epochs 5

    # Real benchmark (TinyStories)
    python train_v2.py --dataset tinystories --size small --epochs 10 --compare

    # WikiText
    python train_v2.py --dataset wikitext --size small --epochs 10 --compare

    # All datasets
    python train_v2.py --dataset all --size small --epochs 10 --compare
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, IterableDataset
import argparse
import time
import json
import os
import math
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
# Datasets
# ============================================================

class ByteDataset(Dataset):
    """Byte-level dataset from raw text."""
    
    def __init__(self, data: torch.Tensor, seq_len: int = 512, stride: int = None):
        self.data = data
        self.seq_len = seq_len
        self.stride = stride or seq_len // 2
    
    def __len__(self):
        return max(0, (len(self.data) - self.seq_len - 1) // self.stride)
    
    def __getitem__(self, idx):
        start = idx * self.stride
        x = self.data[start:start + self.seq_len]
        y = self.data[start + 1:start + self.seq_len + 1]
        return x, y


def text_to_bytes(text: str, max_bytes: int = None) -> torch.Tensor:
    """Convert text to byte tensor, clamping to valid range."""
    raw = text.encode('utf-8', errors='replace')
    if max_bytes:
        raw = raw[:max_bytes]
    return torch.tensor(list(raw), dtype=torch.long)


def load_shakespeare(seq_len=512, max_bytes=None):
    """~1MB Shakespeare."""
    os.makedirs("data", exist_ok=True)
    path = "data/shakespeare.txt"
    if not os.path.exists(path):
        print("📥 Downloading Shakespeare...")
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
            path
        )
    with open(path) as f:
        text = f.read()
    
    data = text_to_bytes(text, max_bytes)
    split = int(0.9 * len(data))
    
    train = ByteDataset(data[:split], seq_len)
    val = ByteDataset(data[split:], seq_len, stride=seq_len)  # non-overlapping val
    print(f"📖 Shakespeare: {len(data):,} bytes | Train: {len(train)} | Val: {len(val)}")
    return train, val


def load_tinystories(seq_len=512, max_bytes=50_000_000):
    """~500MB children's stories. Default: use 50MB subset."""
    print("📥 Loading TinyStories from HuggingFace...")
    from datasets import load_dataset
    
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    
    # Collect text up to max_bytes
    texts = []
    total = 0
    for item in ds:
        t = item["text"]
        texts.append(t)
        total += len(t.encode('utf-8'))
        if total >= max_bytes:
            break
    
    text = "\n\n".join(texts)
    data = text_to_bytes(text, max_bytes)
    
    split = int(0.95 * len(data))
    train = ByteDataset(data[:split], seq_len)
    val = ByteDataset(data[split:], seq_len, stride=seq_len)
    print(f"📖 TinyStories: {len(data):,} bytes ({len(data)/1e6:.1f}MB) | Train: {len(train)} | Val: {len(val)}")
    return train, val


def load_wikitext(seq_len=512, max_bytes=50_000_000):
    """WikiText-103 (~100MB). Default: use 50MB subset."""
    print("📥 Loading WikiText from HuggingFace...")
    from datasets import load_dataset
    
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
    
    texts = []
    total = 0
    for item in ds:
        t = item["text"]
        if len(t.strip()) > 50:  # skip tiny fragments
            texts.append(t)
            total += len(t.encode('utf-8'))
        if total >= max_bytes:
            break
    
    text = "\n".join(texts)
    data = text_to_bytes(text, max_bytes)
    
    split = int(0.95 * len(data))
    train = ByteDataset(data[:split], seq_len)
    val = ByteDataset(data[split:], seq_len, stride=seq_len)
    print(f"📖 WikiText: {len(data):,} bytes ({len(data)/1e6:.1f}MB) | Train: {len(train)} | Val: {len(val)}")
    return train, val


DATASET_LOADERS = {
    "shakespeare": load_shakespeare,
    "tinystories": load_tinystories,
    "wikitext": load_wikitext,
}


# ============================================================
# Fair Baseline (matched params)
# ============================================================

def create_fair_baseline(target_params: int, seq_len: int = 512) -> StandardByteTransformer:
    """Create a standard transformer that matches the target param count."""
    # Binary search for d_model that gives ~target_params
    best = None
    best_diff = float('inf')
    
    for d in [128, 192, 256, 320, 384, 448, 512]:
        for nl in [4, 6, 8, 10, 12]:
            n_heads = max(2, d // 64)
            model = StandardByteTransformer(
                d_model=d, n_heads=n_heads, n_layers=nl,
                d_ff=d*2, max_seq_len=seq_len,
            )
            p = sum(pp.numel() for pp in model.parameters())
            diff = abs(p - target_params)
            if diff < best_diff:
                best_diff = diff
                best = (d, n_heads, nl, d*2, p)
    
    d, nh, nl, dff, p = best
    print(f"  Fair baseline: d={d}, heads={nh}, layers={nl}, ff={dff} → {p:,} params")
    return StandardByteTransformer(d_model=d, n_heads=nh, n_layers=nl, d_ff=dff, max_seq_len=seq_len)


# ============================================================
# Training
# ============================================================

def train_model(model, train_ds, val_ds, device, epochs, batch_size, lr, name, 
                log_every=50, grad_accum=1):
    model = model.to(device)
    params = sum(p.numel() for p in model.parameters())
    
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1, betas=(0.9, 0.95))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
                          num_workers=0, pin_memory=False)
    val_dl = DataLoader(val_ds, batch_size=batch_size, drop_last=True, num_workers=0)
    
    total_steps = len(train_dl) * epochs // grad_accum
    warmup = min(total_steps // 10, 500)
    
    def lr_fn(step):
        if step < warmup:
            return step / max(warmup, 1)
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return max(0.1, 0.5 * (1 + math.cos(math.pi * progress)))
    
    sched = optim.lr_scheduler.LambdaLR(opt, lr_fn)
    
    print(f"\n{'='*60}")
    print(f"Training {name}")
    print(f"  Params: {params:,} | Device: {device} | Batch: {batch_size}")
    print(f"  Steps/epoch: {len(train_dl)} | Grad accum: {grad_accum}")
    print(f"{'='*60}")
    
    history = {"train_loss": [], "val_loss": [], "val_bpb": [], "epoch_time": [], "gate_dynamics": []}
    best_val = float('inf')
    step = 0
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        n = 0
        t0 = time.time()
        opt.zero_grad()
        
        for batch_idx, (x, y) in enumerate(train_dl):
            x, y = x.to(device), y.to(device)
            
            out = model(x, targets=y)
            loss = out["loss"] / grad_accum
            loss.backward()
            
            if (batch_idx + 1) % grad_accum == 0:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                step += 1
            
            epoch_loss += out["loss"].item()
            n += 1
            
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
                print(f"  [{name:>14}] Ep {epoch+1} Step {batch_idx+1}/{len(train_dl)} | "
                      f"Loss {avg:.4f} BPB {bpb:.3f} LR {lr_now:.1e}{extra}")
        
        dt = time.time() - t0
        train_loss = epoch_loss / max(n, 1)
        
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
        
        # Gate dynamics
        if hasattr(model, 'get_gate_stats'):
            gates = model.get_gate_stats()
            if gates:
                avg_g = {"P": 0, "I": 0, "D": 0}
                for g in gates.values():
                    for k in avg_g:
                        avg_g[k] += g[k]
                for k in avg_g:
                    avg_g[k] /= len(gates)
                history["gate_dynamics"].append(avg_g)
        
        if val_loss < best_val:
            best_val = val_loss
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), f"checkpoints/{name}_best.pt")
        
        print(f"  [{name:>14}] Epoch {epoch+1}/{epochs} | "
              f"Train {train_loss:.4f} | Val {val_loss:.4f} | BPB {val_bpb:.3f} | {dt:.1f}s")
    
    history["best_val"] = best_val
    history["best_bpb"] = best_val / math.log(2)
    history["params"] = params
    return history


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="ionBrain Training v2")
    parser.add_argument("--dataset", default="tinystories", 
                        choices=["shakespeare", "tinystories", "wikitext", "all"])
    parser.add_argument("--compare", action="store_true", help="Run fair baseline comparison")
    parser.add_argument("--size", default="small", choices=["tiny", "small", "base"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--max-mb", type=int, default=50, help="Max dataset size in MB")
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--generate", action="store_true")
    args = parser.parse_args()
    
    device = get_device()
    max_bytes = args.max_mb * 1_000_000
    
    datasets_to_run = [args.dataset] if args.dataset != "all" else ["shakespeare", "tinystories", "wikitext"]
    
    all_results = {}
    
    for ds_name in datasets_to_run:
        print(f"\n{'#'*60}")
        print(f"# Dataset: {ds_name}")
        print(f"{'#'*60}")
        
        loader = DATASET_LOADERS[ds_name]
        if ds_name == "shakespeare":
            train_ds, val_ds = loader(args.seq_len)
        else:
            train_ds, val_ds = loader(args.seq_len, max_bytes)
        
        results = {}
        
        # ionBrain
        model = create_ionbrain(args.size)
        ion_params = sum(p.numel() for p in model.parameters())
        h = train_model(model, train_ds, val_ds, device, args.epochs, args.batch_size, 
                        args.lr, f"ionBrain-{args.size}", grad_accum=args.grad_accum)
        results[f"ionBrain-{args.size}"] = h
        
        if args.generate and hasattr(model, 'generate'):
            model.eval()
            prompt = "Once upon a time" if ds_name == "tinystories" else "To be or not"
            x = torch.tensor([[ord(c) for c in prompt]], dtype=torch.long, device=device)
            with torch.no_grad():
                gen = model.generate(x, max_new=200, temperature=0.8)
            text = ''.join(chr(min(b, 127)) for b in gen[0].tolist())
            print(f"\n📝 ionBrain generation:\n{'─'*40}\n{text}\n{'─'*40}")
        
        # Gate analysis
        if hasattr(model, 'get_gate_stats'):
            print(f"\n🔍 Gate Analysis:")
            model.eval()
            with torch.no_grad():
                x = torch.tensor([[ord(c) for c in "The meaning of life"]], dtype=torch.long, device=device)
                _ = model(x)
            for name, g in model.get_gate_stats().items():
                dom = max(g, key=g.get)
                print(f"  {name}: P={g['P']:.3f} I={g['I']:.3f} D={g['D']:.3f} → {dom}")
            
            if h.get("gate_dynamics"):
                print(f"\n  Gate evolution over training:")
                for i, g in enumerate(h["gate_dynamics"]):
                    print(f"    Ep {i+1}: P={g['P']:.3f} I={g['I']:.3f} D={g['D']:.3f}")
        
        # Fair baseline
        if args.compare:
            print(f"\n🎯 Creating fair baseline (~{ion_params:,} params)...")
            baseline = create_fair_baseline(ion_params, args.seq_len)
            h2 = train_model(baseline, train_ds, val_ds, device, args.epochs, args.batch_size,
                            args.lr, "Std-ByteTF-Fair", grad_accum=args.grad_accum)
            results["Std-ByteTF-Fair"] = h2
        
        # Summary
        print(f"\n{'='*60}")
        print(f"RESULTS — {ds_name}")
        print(f"{'='*60}")
        print(f"{'Model':<20} {'Params':>10} {'Best BPB':>10} {'Best Loss':>12} {'Ep Time':>10}")
        print("-" * 64)
        for name, h in results.items():
            avg_time = sum(h["epoch_time"]) / len(h["epoch_time"])
            print(f"{name:<20} {h['params']:>10,} {h['best_bpb']:>10.3f} {h['best_val']:>12.4f} {avg_time:>9.1f}s")
        
        if len(results) > 1:
            names = list(results.keys())
            bpb1 = results[names[0]]["best_bpb"]
            bpb2 = results[names[1]]["best_bpb"]
            if bpb1 < bpb2:
                print(f"\n🏆 {names[0]} wins! BPB {bpb1:.3f} vs {bpb2:.3f} ({(1-bpb1/bpb2)*100:.1f}% better)")
            else:
                print(f"\n📊 {names[1]} wins. BPB {bpb2:.3f} vs {bpb1:.3f} ({(1-bpb2/bpb1)*100:.1f}% better)")
        
        all_results[ds_name] = results
    
    # Save all results
    Path("results").mkdir(exist_ok=True)
    save = {}
    for ds, results in all_results.items():
        save[ds] = {}
        for name, h in results.items():
            save[ds][name] = {
                "params": h["params"],
                "best_val": h["best_val"],
                "best_bpb": h["best_bpb"],
                "train_loss": h["train_loss"],
                "val_loss": h["val_loss"],
                "val_bpb": h["val_bpb"],
                "epoch_time": h["epoch_time"],
                "gate_dynamics": h.get("gate_dynamics", []),
            }
    
    with open("results/v2_results.json", "w") as f:
        json.dump(save, f, indent=2, default=str)
    
    print(f"\n✅ All results saved to results/v2_results.json")


if __name__ == "__main__":
    main()
