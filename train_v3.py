"""
ionBrain v3 Training — Scheduled Sampling + Anti-Collapse

The key innovation: during training, we progressively replace teacher-forced
inputs with the model's own predictions, forcing the PID controller to learn
self-correction on noisy inputs (not just clean ground truth).

Usage:
    python train_v3.py --dataset tinystories --size small --epochs 10 --compare --generate
    python train_v3.py --dataset shakespeare --size small --epochs 20 --generate
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import argparse
import time
import json
import os
import math
import urllib.request
from pathlib import Path

from ionbrain_v3 import ionBrainV3, StandardByteTransformer, create_ionbrain_v3


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
# Datasets (same as v2)
# ============================================================

class ByteDataset(Dataset):
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


def text_to_bytes(text, max_bytes=None):
    raw = text.encode('utf-8', errors='replace')
    if max_bytes:
        raw = raw[:max_bytes]
    return torch.tensor(list(raw), dtype=torch.long)


def load_shakespeare(seq_len=512, max_bytes=None):
    os.makedirs("data", exist_ok=True)
    path = "data/shakespeare.txt"
    if not os.path.exists(path):
        print("📥 Downloading Shakespeare...")
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt", path)
    with open(path) as f:
        text = f.read()
    data = text_to_bytes(text, max_bytes)
    split = int(0.9 * len(data))
    train = ByteDataset(data[:split], seq_len)
    val = ByteDataset(data[split:], seq_len, stride=seq_len)
    print(f"📖 Shakespeare: {len(data):,} bytes | Train: {len(train)} | Val: {len(val)}")
    return train, val


def load_tinystories(seq_len=512, max_bytes=50_000_000):
    print("📥 Loading TinyStories from HuggingFace...")
    from datasets import load_dataset
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    texts = []
    total = 0
    for item in ds:
        texts.append(item["text"])
        total += len(item["text"].encode('utf-8'))
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
    print("📥 Loading WikiText from HuggingFace...")
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
    texts = []
    total = 0
    for item in ds:
        if len(item["text"].strip()) > 50:
            texts.append(item["text"])
            total += len(item["text"].encode('utf-8'))
        if total >= max_bytes:
            break
    text = "\n".join(texts)
    data = text_to_bytes(text, max_bytes)
    split = int(0.95 * len(data))
    train = ByteDataset(data[:split], seq_len)
    val = ByteDataset(data[split:], seq_len, stride=seq_len)
    print(f"📖 WikiText: {len(data):,} bytes ({len(data)/1e6:.1f}MB) | Train: {len(train)} | Val: {len(val)}")
    return train, val


LOADERS = {"shakespeare": load_shakespeare, "tinystories": load_tinystories, "wikitext": load_wikitext}


# ============================================================
# Scheduled Sampling
# ============================================================

def get_sampling_probability(epoch: int, total_epochs: int, warmup_epochs: int = 2, 
                              max_prob: float = 0.5) -> float:
    """
    Curriculum for scheduled sampling.
    
    - Epochs 0 to warmup_epochs: 0% (pure teacher forcing, let PID learn basics)
    - warmup_epochs to total_epochs: linearly increase to max_prob
    
    The model gradually sees more of its own predictions during training.
    """
    if epoch < warmup_epochs:
        return 0.0
    progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
    return min(max_prob, progress * max_prob)


def create_mixed_input(model, x: torch.Tensor, y: torch.Tensor, 
                       sample_prob: float, device: torch.device) -> torch.Tensor:
    """
    Create mixed input for scheduled sampling.
    
    For each position, with probability sample_prob, replace the ground truth
    input with the model's own prediction from the previous position.
    
    This is the key mechanism that forces the PID to handle noisy/wrong inputs.
    """
    if sample_prob <= 0:
        return x
    
    B, T = x.shape
    
    # Get model predictions (no grad — we're just sampling, not backprop through this)
    with torch.no_grad():
        out = model(x)
        logits = out["logits"]
        # Sample from predictions (not argmax — maintain diversity)
        probs = torch.softmax(logits / 0.8, dim=-1)
        predicted = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(B, T)
    
    # Create sampling mask: which positions use model predictions
    mask = torch.bernoulli(torch.full((B, T), sample_prob, device=device)).bool()
    
    # Never replace position 0 (need a real start token)
    mask[:, 0] = False
    
    # For sampled positions, use the PREVIOUS position's prediction as input
    # (shifted by 1 because input at position t should predict position t)
    mixed = x.clone()
    if T > 1:
        # predicted[:, t-1] is what the model predicted for position t
        shifted_pred = torch.cat([x[:, :1], predicted[:, :-1]], dim=1)
        mixed[mask] = shifted_pred[mask]
    
    return mixed


# ============================================================
# Fair Baseline
# ============================================================

def create_fair_baseline(target_params, seq_len=512):
    best = None
    best_diff = float('inf')
    for d in [128, 192, 256, 320, 384, 448, 512]:
        for nl in [4, 6, 8, 10, 12]:
            nh = max(2, d // 64)
            m = StandardByteTransformer(d_model=d, n_heads=nh, n_layers=nl, d_ff=d*2, max_seq_len=seq_len)
            p = sum(pp.numel() for pp in m.parameters())
            if abs(p - target_params) < best_diff:
                best_diff = abs(p - target_params)
                best = (d, nh, nl, d*2, p)
    d, nh, nl, dff, p = best
    print(f"  Fair baseline: d={d}, heads={nh}, layers={nl} → {p:,} params")
    return StandardByteTransformer(d_model=d, n_heads=nh, n_layers=nl, d_ff=dff, max_seq_len=seq_len)


# ============================================================
# Generation Test
# ============================================================

def test_generation(model, device, prompts=None, max_new=150, temp=0.8):
    """Test generation quality with multiple prompts."""
    if prompts is None:
        prompts = [
            "Once upon a time",
            "The king said",
            "1 + 1 = ",
            "ABCDEFG",
            "She walked into the",
        ]
    
    model.eval()
    results = []
    for prompt in prompts:
        px = torch.tensor([[ord(c) for c in prompt]], device=device)
        with torch.no_grad():
            gen = model.generate(px, max_new=max_new, temperature=temp, top_k=40)
        text = "".join(chr(min(b, 127)) for b in gen[0].tolist())
        results.append((prompt, text))
        
        # Check for repetition collapse
        output_part = text[len(prompt):]
        if len(output_part) > 10:
            # Count max consecutive same char
            max_repeat = 1
            current = 1
            for i in range(1, len(output_part)):
                if output_part[i] == output_part[i-1]:
                    current += 1
                    max_repeat = max(max_repeat, current)
                else:
                    current = 1
            
            collapse = "🔴 COLLAPSE" if max_repeat > 5 else "🟢 OK" if max_repeat <= 3 else "🟡 WARN"
            print(f"  📝 [{collapse}] {prompt}")
            print(f"     → {text[:120]}")
        else:
            print(f"  📝 {prompt} → {text[:120]}")
    
    return results


# ============================================================
# Training
# ============================================================

def train_model(model, train_ds, val_ds, device, epochs, batch_size, lr, name, 
                log_every=50, use_scheduled_sampling=False):
    model = model.to(device)
    params = sum(p.numel() for p in model.parameters())
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1, betas=(0.9, 0.95))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, drop_last=True)
    
    total_steps = len(train_dl) * epochs
    warmup = min(total_steps // 10, 500)
    
    def lr_fn(step):
        if step < warmup:
            return step / max(warmup, 1)
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return max(0.1, 0.5 * (1 + math.cos(math.pi * progress)))
    
    sched = optim.lr_scheduler.LambdaLR(opt, lr_fn)
    
    print(f"\n{'='*60}")
    print(f"Training {name} | {params:,} params | {device}")
    print(f"  Steps/epoch: {len(train_dl)} | Epochs: {epochs}")
    if use_scheduled_sampling:
        print(f"  🔄 Scheduled sampling ENABLED (warmup=2 epochs, max=50%)")
    print(f"{'='*60}")
    
    history = {
        "train_loss": [], "val_loss": [], "val_bpb": [], 
        "epoch_time": [], "gate_dynamics": [], "sample_prob": [],
        "stagnation_rates": [],
    }
    best_val = float('inf')
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        n = 0
        t0 = time.time()
        
        # Scheduled sampling probability for this epoch
        sample_prob = get_sampling_probability(epoch, epochs) if use_scheduled_sampling else 0.0
        
        for batch_idx, (x, y) in enumerate(train_dl):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            
            # Scheduled sampling: create mixed input
            if sample_prob > 0 and isinstance(model, ionBrainV3):
                mixed = create_mixed_input(model, x, y, sample_prob, device)
                out = model(x, targets=y, mixed_input=mixed)
            else:
                out = model(x, targets=y)
            
            out["loss"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            
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
                
                rep = f" | Rep: {out.get('rep_rate', 0):.2f}" if 'rep_rate' in out else ""
                ss = f" | SS: {sample_prob:.0%}" if sample_prob > 0 else ""
                
                # Stagnation info
                stag = ""
                if hasattr(model, 'get_stagnation_stats'):
                    stag_stats = model.get_stagnation_stats()
                    stag_vals = [v for v in stag_stats.values() if v > 0]
                    if stag_vals:
                        stag = f" | Stag: {sum(stag_vals)/len(stag_vals):.2f}"
                
                print(f"  [{name:>14}] Ep {epoch+1} Step {batch_idx+1}/{len(train_dl)} | "
                      f"Loss {avg:.4f} BPB {bpb:.3f} LR {lr_now:.1e}{extra}{rep}{ss}{stag}")
        
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
        history["sample_prob"].append(sample_prob)
        
        if hasattr(model, 'get_gate_stats'):
            gates = model.get_gate_stats()
            if gates:
                avg_g = {"P": 0, "I": 0, "D": 0}
                for g in gates.values():
                    for k in avg_g: avg_g[k] += g[k]
                for k in avg_g: avg_g[k] /= len(gates)
                history["gate_dynamics"].append(avg_g)
        
        if hasattr(model, 'get_stagnation_stats'):
            stag_stats = model.get_stagnation_stats()
            stag_vals = [v for v in stag_stats.values() if v > 0]
            history["stagnation_rates"].append(
                sum(stag_vals)/len(stag_vals) if stag_vals else 0.0
            )
        
        if val_loss < best_val:
            best_val = val_loss
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), f"checkpoints/{name}_best.pt")
        
        ss_str = f" | SS: {sample_prob:.0%}" if sample_prob > 0 else ""
        print(f"  [{name:>14}] Epoch {epoch+1}/{epochs} | "
              f"Train {train_loss:.4f} | Val {val_loss:.4f} | BPB {val_bpb:.3f} | {dt:.1f}s{ss_str}")
        
        # Generation test every epoch — THE critical check
        if hasattr(model, 'generate'):
            print(f"\n  --- Epoch {epoch+1} Generation Check ---")
            test_generation(model, device, max_new=80)
            print()
    
    history["best_val"] = best_val
    history["best_bpb"] = best_val / math.log(2)
    history["params"] = params
    return history


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="tinystories", choices=["shakespeare", "tinystories", "wikitext"])
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--size", default="small", choices=["tiny", "small", "base"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--max-mb", type=int, default=50)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--no-scheduled-sampling", action="store_true", 
                        help="Disable scheduled sampling (for ablation)")
    args = parser.parse_args()
    
    device = get_device()
    max_bytes = args.max_mb * 1_000_000
    
    loader = LOADERS[args.dataset]
    if args.dataset == "shakespeare":
        train_ds, val_ds = loader(args.seq_len)
    else:
        train_ds, val_ds = loader(args.seq_len, max_bytes)
    
    results = {}
    use_ss = not args.no_scheduled_sampling
    
    # ionBrain v3
    model = create_ionbrain_v3(args.size)
    ion_params = sum(p.numel() for p in model.parameters())
    counts = model.count_params()
    print(f"\n📊 ionBrain v3 params: {counts}")
    
    h = train_model(model, train_ds, val_ds, device, args.epochs, args.batch_size, args.lr, 
                    f"ionBrain-v3-{args.size}", use_scheduled_sampling=use_ss)
    results[f"ionBrain-v3-{args.size}"] = h
    
    if args.generate and hasattr(model, 'generate'):
        print(f"\n{'='*60}")
        print("FINAL GENERATION TEST")
        print(f"{'='*60}")
        test_generation(model, device, max_new=200)
    
    # Gate evolution
    if hasattr(model, 'get_gate_stats') and h.get("gate_dynamics"):
        print(f"\n🔍 Gate Evolution:")
        for i, g in enumerate(h["gate_dynamics"]):
            ss = h["sample_prob"][i] if i < len(h["sample_prob"]) else 0
            stag = h["stagnation_rates"][i] if i < len(h.get("stagnation_rates", [])) else 0
            print(f"  Ep {i+1}: P={g['P']:.3f} I={g['I']:.3f} D={g['D']:.3f} | SS={ss:.0%} | Stag={stag:.3f}")
    
    # Baseline
    if args.compare:
        print(f"\n🎯 Fair baseline (~{ion_params:,} params)...")
        baseline = create_fair_baseline(ion_params, args.seq_len)
        h2 = train_model(baseline, train_ds, val_ds, device, args.epochs, args.batch_size, args.lr, "Std-ByteTF")
        results["Std-ByteTF"] = h2
    
    # Summary
    print(f"\n{'='*60}")
    print(f"RESULTS — {args.dataset} (v3 {'with' if use_ss else 'WITHOUT'} scheduled sampling)")
    print(f"{'='*60}")
    print(f"{'Model':<22} {'Params':>10} {'Best BPB':>10} {'Best Loss':>12}")
    print("-" * 56)
    for name, h in results.items():
        print(f"{name:<22} {h['params']:>10,} {h['best_bpb']:>10.3f} {h['best_val']:>12.4f}")
    
    if len(results) > 1:
        names = list(results.keys())
        b1, b2 = results[names[0]]["best_bpb"], results[names[1]]["best_bpb"]
        winner = names[0] if b1 < b2 else names[1]
        loser_bpb = max(b1, b2)
        winner_bpb = min(b1, b2)
        print(f"\n🏆 {winner} wins! BPB {winner_bpb:.3f} vs {loser_bpb:.3f} ({(1-winner_bpb/loser_bpb)*100:.1f}% better)")
    
    Path("results").mkdir(exist_ok=True)
    with open("results/v3_results.json", "w") as f:
        json.dump({n: {"params": h["params"], "best_bpb": h["best_bpb"], "best_val": h["best_val"],
                       "val_bpb": h["val_bpb"], "gate_dynamics": h.get("gate_dynamics", []),
                       "sample_prob": h.get("sample_prob", []),
                       "stagnation_rates": h.get("stagnation_rates", [])}
                   for n, h in results.items()}, f, indent=2)
    print(f"\n✅ Saved to results/v3_results.json")


if __name__ == "__main__":
    main()
