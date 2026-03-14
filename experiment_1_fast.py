"""
Experiment 1 (Fast) — smaller scale to run quickly on CPU.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
import json
from pathlib import Path
from pid_layer import PIDLinear, PIDNet, pid_sparsity_loss


def generate_pid_data(n_samples, seq_len, d_input, pid_gains=None, seed=42):
    rng = np.random.RandomState(seed)
    
    if pid_gains is None:
        pid_gains = {
            0: {"Kp": 1.0, "Ki": 0.5, "Kd": 0.1},   # P-dominant
            1: {"Kp": 0.1, "Ki": 1.0, "Kd": 0.2},   # I-dominant
            2: {"Kp": 0.2, "Ki": 0.1, "Kd": 1.0},   # D-dominant
            3: {"Kp": 0.5, "Ki": 0.5, "Kd": 0.5},   # Balanced
        }
    
    X = np.zeros((n_samples, seq_len, d_input), dtype=np.float32)
    for i in range(n_samples):
        for ch in range(d_input):
            t = np.linspace(0, 4 * np.pi, seq_len)
            sig_type = i % 4
            if sig_type == 0:
                X[i, :, ch] = np.sin(rng.uniform(0.5, 3.0) * t)
            elif sig_type == 1:
                step = rng.randint(seq_len//4, 3*seq_len//4)
                X[i, step:, ch] = rng.uniform(0.5, 2.0)
            elif sig_type == 2:
                X[i, :, ch] = rng.uniform(-1, 1) * np.linspace(0, 1, seq_len)
            else:
                X[i, :, ch] = np.cumsum(rng.randn(seq_len) * 0.1)
    
    Y = np.zeros((n_samples, seq_len, 1), dtype=np.float32)
    alpha = 0.9
    
    for ch in range(d_input):
        Kp, Ki, Kd = pid_gains[ch]["Kp"], pid_gains[ch]["Ki"], pid_gains[ch]["Kd"]
        x = X[:, :, ch]
        
        # P
        Y[:, :, 0] += Kp * x
        
        # I (EMA)
        integral = np.zeros_like(x)
        for t in range(1, seq_len):
            integral[:, t] = alpha * integral[:, t-1] + (1 - alpha) * x[:, t]
        Y[:, :, 0] += Ki * integral
        
        # D
        deriv = np.zeros_like(x)
        deriv[:, 1:] = x[:, 1:] - x[:, :-1]
        Y[:, :, 0] += Kd * deriv
    
    Y += rng.randn(*Y.shape).astype(np.float32) * 0.01
    return torch.from_numpy(X), torch.from_numpy(Y), pid_gains


class MLPBaseline(nn.Module):
    def __init__(self, d_in, d_hid, d_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hid), nn.GELU(),
            nn.Linear(d_hid, d_hid), nn.GELU(),
            nn.Linear(d_hid, d_out)
        )
    def forward(self, x):
        return self.net(x)


class LSTMBaseline(nn.Module):
    def __init__(self, d_in, d_hid, d_out):
        super().__init__()
        self.lstm = nn.LSTM(d_in, d_hid, 2, batch_first=True)
        self.proj = nn.Linear(d_hid, d_out)
    def forward(self, x):
        h, _ = self.lstm(x)
        return self.proj(h)


def train(model, train_dl, val_dl, epochs, lr, name, sparsity_w=0.0):
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.MSELoss()
    
    best_val = float('inf')
    target_epoch = None
    
    for ep in range(epochs):
        model.train()
        t_loss = 0
        t0 = time.time()
        for xb, yb in train_dl:
            opt.zero_grad()
            loss = crit(model(xb), yb)
            if sparsity_w > 0:
                loss = loss + sparsity_w * pid_sparsity_loss(model)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            t_loss += loss.item()
        sched.step()
        t_loss /= len(train_dl)
        
        model.eval()
        v_loss = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                v_loss += crit(model(xb), yb).item()
        v_loss /= len(val_dl)
        
        if v_loss < best_val:
            best_val = v_loss
        if v_loss < 0.01 and target_epoch is None:
            target_epoch = ep + 1
        
        if (ep+1) % 10 == 0 or ep == 0:
            dt = time.time() - t0
            print(f"  [{name:>8}] Ep {ep+1:3d} | Train {t_loss:.6f} | Val {v_loss:.6f} | {dt:.1f}s")
    
    params = sum(p.numel() for p in model.parameters())
    return {"best_val": best_val, "target_epoch": target_epoch, "params": params}


def main():
    print("=" * 60)
    print("EXPERIMENT 1: Synthetic PID Recovery (Fast)")
    print("=" * 60)
    
    D, H, SEQ = 4, 32, 32
    
    print("\n📊 Generating data...")
    X_tr, Y_tr, gains = generate_pid_data(2000, SEQ, D, seed=42)
    X_va, Y_va, _ = generate_pid_data(500, SEQ, D, seed=99)
    
    tr_dl = DataLoader(TensorDataset(X_tr, Y_tr), batch_size=64, shuffle=True)
    va_dl = DataLoader(TensorDataset(X_va, Y_va), batch_size=64)
    
    print(f"  Train: {X_tr.shape}, Val: {X_va.shape}")
    
    print("\nTrue PID gains:")
    for ch, g in gains.items():
        total = g["Kp"] + g["Ki"] + g["Kd"]
        print(f"  Ch{ch}: P={g['Kp']:.1f}({g['Kp']/total:.0%}) "
              f"I={g['Ki']:.1f}({g['Ki']/total:.0%}) "
              f"D={g['Kd']:.1f}({g['Kd']/total:.0%})")
    
    EPOCHS = 60
    LR = 3e-3
    
    models = {
        "MLP": MLPBaseline(D, H, 1),
        "LSTM": LSTMBaseline(D, H, 1),
        "PID-Net": PIDNet(d_input=D, d_model=H, d_output=1, n_layers=2),
    }
    
    print(f"\n📐 Params: " + " | ".join(
        f"{n}: {sum(p.numel() for p in m.parameters()):,}" 
        for n, m in models.items()
    ))
    
    results = {}
    for name, model in models.items():
        print(f"\n🏋️ Training {name}...")
        sw = 0.01 if "PID" in name else 0
        results[name] = train(model, tr_dl, va_dl, EPOCHS, LR, name, sw)
    
    # === Results ===
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\n{'Model':<12} {'Params':>8} {'Best Val MSE':>14} {'Epochs→0.01':>14}")
    print("-" * 50)
    for n, r in results.items():
        te = str(r['target_epoch']) if r['target_epoch'] else "N/A"
        print(f"{n:<12} {r['params']:>8,} {r['best_val']:>14.6f} {te:>14}")
    
    # === Gate Analysis ===
    pid = models["PID-Net"]
    pid.eval()
    print("\n🔍 PID-Net Gate Analysis:")
    with torch.no_grad():
        _ = pid(X_va[:32])
    stats = pid.get_all_gate_stats()
    for layer_name, s in stats.items():
        if s:
            print(f"  {layer_name}: P={s['gate_p_mean']:.3f} I={s['gate_i_mean']:.3f} D={s['gate_d_mean']:.3f}")
    
    # Alpha/beta
    for name, mod in pid.named_modules():
        if isinstance(mod, PIDLinear):
            print(f"\n  [{name}] α={mod.alpha.mean():.3f} (true=0.9) | β={mod.beta.mean():.3f}")
    
    # === Verdict ===
    print("\n" + "=" * 60)
    pid_v = results["PID-Net"]["best_val"]
    mlp_v = results["MLP"]["best_val"]
    lstm_v = results["LSTM"]["best_val"]
    
    if pid_v < mlp_v and pid_v < lstm_v:
        print(f"🏆 PID-Net WINS! {pid_v:.6f} vs MLP {mlp_v:.6f} vs LSTM {lstm_v:.6f}")
        print(f"   vs MLP:  {(1-pid_v/mlp_v)*100:.1f}% better")
        print(f"   vs LSTM: {(1-pid_v/lstm_v)*100:.1f}% better")
    elif pid_v < mlp_v:
        print(f"✅ PID-Net beats MLP ({pid_v:.6f} vs {mlp_v:.6f}), LSTM slightly ahead ({lstm_v:.6f})")
    else:
        print(f"⚠️ Need tuning — PID: {pid_v:.6f} | MLP: {mlp_v:.6f} | LSTM: {lstm_v:.6f}")
    print("=" * 60)
    
    # Save
    Path("results").mkdir(exist_ok=True)
    with open("results/exp1_fast.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("✅ Saved to results/exp1_fast.json")


if __name__ == "__main__":
    main()
