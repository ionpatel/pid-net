"""
Experiment 1 — Minimal version with vectorized PID.
No sequential loops — pure tensor ops.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time, math


class FastPIDLinear(nn.Module):
    """PID layer with vectorized integral/derivative (no python loops)."""
    
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_p = nn.Parameter(torch.empty(d_out, d_in))
        self.W_i = nn.Parameter(torch.empty(d_out, d_in))
        self.W_d = nn.Parameter(torch.empty(d_out, d_in))
        self.gate = nn.Linear(3 * d_in, 3)
        self.bias = nn.Parameter(torch.zeros(d_out))
        
        nn.init.kaiming_uniform_(self.W_p, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_i, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_d, a=math.sqrt(5))
        self.W_i.data *= 0.1
        self.W_d.data *= 0.1
        self.gate.bias.data[0] = 1.0  # favor P initially
        
        self._alpha = nn.Parameter(torch.tensor(2.2))  # sigmoid → ~0.9
    
    @property
    def alpha(self):
        return torch.sigmoid(self._alpha)
    
    def forward(self, x):
        """x: (B, T, D)"""
        B, T, D = x.shape
        
        # Vectorized integral: cumulative weighted sum
        # S(t) ≈ Σ α^(t-k) * (1-α) * x(k)  — geometric weighting
        a = self.alpha
        weights = (1 - a) * a.pow(torch.arange(T, device=x.device).float().flip(0))
        # weights[k] = (1-α) * α^(T-1-k) — weight for position k when computing S(T-1)
        # Use cumsum trick: for each t, S(t) = cumsum of weighted x up to t
        # Actually, use conv1d for efficiency
        
        # Simpler: just use shifted cumulative mean as proxy for integral
        cumsum = torch.cumsum(x, dim=1)
        counts = torch.arange(1, T+1, device=x.device).float().view(1, T, 1)
        integral = cumsum / counts  # running mean — good proxy for soft integral
        
        # Derivative: finite difference, padded
        derivative = torch.zeros_like(x)
        derivative[:, 1:, :] = x[:, 1:, :] - x[:, :-1, :]
        
        # Gates
        gate_in = torch.cat([x, integral, derivative], dim=-1)
        g = F.softmax(self.gate(gate_in), dim=-1)  # (B, T, 3)
        
        p_out = F.linear(x, self.W_p)
        i_out = F.linear(integral, self.W_i)
        d_out = F.linear(derivative, self.W_d)
        
        y = (g[:,:,0:1] * p_out + g[:,:,1:2] * i_out + g[:,:,2:3] * d_out + self.bias)
        
        self._last_gates = g.detach()
        return y


class FastPIDNet(nn.Module):
    def __init__(self, d_in, d_model, d_out, n_layers=2):
        super().__init__()
        self.proj_in = nn.Linear(d_in, d_model)
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.LayerNorm(d_model))
            self.layers.append(FastPIDLinear(d_model, d_model * 2))
            self.layers.append(nn.GELU())
            self.layers.append(FastPIDLinear(d_model * 2, d_model))
        self.norm = nn.LayerNorm(d_model)
        self.proj_out = nn.Linear(d_model, d_out)
    
    def forward(self, x):
        x = self.proj_in(x)
        for i in range(0, len(self.layers), 4):
            residual = x
            x = self.layers[i](x)      # norm
            x = self.layers[i+1](x)     # pid1
            x = self.layers[i+2](x)     # gelu
            x = self.layers[i+3](x)     # pid2
            x = x + residual
        return self.proj_out(self.norm(x))
    
    def get_gate_stats(self):
        stats = []
        for m in self.modules():
            if isinstance(m, FastPIDLinear) and hasattr(m, '_last_gates'):
                g = m._last_gates
                stats.append({
                    "P": g[:,:,0].mean().item(),
                    "I": g[:,:,1].mean().item(),
                    "D": g[:,:,2].mean().item(),
                })
        return stats


class MLP(nn.Module):
    def __init__(self, d_in, d_hid, d_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hid), nn.GELU(),
            nn.Linear(d_hid, d_hid), nn.GELU(),
            nn.Linear(d_hid, d_out))
    def forward(self, x): return self.net(x)


class LSTM(nn.Module):
    def __init__(self, d_in, d_hid, d_out):
        super().__init__()
        self.lstm = nn.LSTM(d_in, d_hid, 2, batch_first=True)
        self.proj = nn.Linear(d_hid, d_out)
    def forward(self, x):
        h, _ = self.lstm(x)
        return self.proj(h)


def make_data(n, T, D, seed):
    rng = np.random.RandomState(seed)
    gains = {
        0: (1.0, 0.5, 0.1),  # P-dominant
        1: (0.1, 1.0, 0.2),  # I-dominant
        2: (0.2, 0.1, 1.0),  # D-dominant
        3: (0.5, 0.5, 0.5),  # balanced
    }
    
    X = rng.randn(n, T, D).astype(np.float32) * 0.5
    # Add structure
    for i in range(n):
        t = np.linspace(0, 4*np.pi, T)
        for ch in range(D):
            if i % 3 == 0:
                X[i, :, ch] += np.sin(rng.uniform(1,3) * t)
            elif i % 3 == 1:
                X[i, :, ch] = np.cumsum(X[i, :, ch]) * 0.3
    
    Y = np.zeros((n, T, 1), dtype=np.float32)
    for ch in range(D):
        Kp, Ki, Kd = gains[ch]
        x = X[:, :, ch]
        Y[:, :, 0] += Kp * x
        # Integral (running mean proxy)
        Y[:, :, 0] += Ki * np.cumsum(x, axis=1) / np.arange(1, T+1)
        # Derivative
        d = np.zeros_like(x)
        d[:, 1:] = x[:, 1:] - x[:, :-1]
        Y[:, :, 0] += Kd * d
    
    Y += rng.randn(*Y.shape).astype(np.float32) * 0.01
    return torch.from_numpy(X), torch.from_numpy(Y), gains


def train_model(model, X_tr, Y_tr, X_va, Y_va, epochs, lr, name):
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.MSELoss()
    best = float('inf')
    target_ep = None
    
    for ep in range(epochs):
        model.train()
        t0 = time.time()
        
        # Full batch (small data)
        opt.zero_grad()
        loss = crit(model(X_tr), Y_tr)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        
        model.eval()
        with torch.no_grad():
            val_loss = crit(model(X_va), Y_va).item()
        
        if val_loss < best:
            best = val_loss
        if val_loss < 0.01 and target_ep is None:
            target_ep = ep + 1
        
        dt = time.time() - t0
        if (ep+1) % 20 == 0 or ep == 0:
            print(f"  [{name:>7}] Ep {ep+1:3d} | Train {loss.item():.6f} | Val {val_loss:.6f} | {dt*1000:.0f}ms")
    
    return best, target_ep


def main():
    print("=" * 60)
    print("EXPERIMENT 1: PID Recovery (Vectorized)")
    print("=" * 60)
    
    D, H, T = 4, 32, 32
    
    X_tr, Y_tr, gains = make_data(1000, T, D, 42)
    X_va, Y_va, _ = make_data(200, T, D, 99)
    
    print(f"\nData: train {X_tr.shape} val {X_va.shape}")
    print(f"True gains: Ch0=(P=1,I=0.5,D=0.1) Ch1=(P=0.1,I=1,D=0.2) "
          f"Ch2=(P=0.2,I=0.1,D=1) Ch3=(P=0.5,I=0.5,D=0.5)")
    
    models = {
        "MLP": MLP(D, H, 1),
        "LSTM": LSTM(D, H, 1),
        "PID-Net": FastPIDNet(D, H, 1, n_layers=2),
    }
    
    for n, m in models.items():
        print(f"  {n}: {sum(p.numel() for p in m.parameters()):,} params")
    
    EPOCHS = 200
    results = {}
    
    for name, model in models.items():
        print(f"\n🏋️ {name}")
        best, target = train_model(model, X_tr, Y_tr, X_va, Y_va, EPOCHS, 3e-3, name)
        results[name] = (best, target)
    
    # Results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"{'Model':<12} {'Best MSE':>12} {'Epochs→0.01':>14}")
    print("-" * 40)
    for n, (b, t) in results.items():
        print(f"{n:<12} {b:>12.6f} {str(t) if t else 'N/A':>14}")
    
    # Gate analysis
    pid = models["PID-Net"]
    pid.eval()
    with torch.no_grad():
        _ = pid(X_va)
    print("\n🔍 Gate Analysis:")
    for i, s in enumerate(pid.get_gate_stats()):
        print(f"  Layer {i}: P={s['P']:.3f} I={s['I']:.3f} D={s['D']:.3f}")
    
    # Verdict
    pid_v = results["PID-Net"][0]
    mlp_v = results["MLP"][0]
    lstm_v = results["LSTM"][0]
    
    print()
    if pid_v < mlp_v and pid_v < lstm_v:
        print(f"🏆 PID-Net WINS! {pid_v:.6f} vs MLP {mlp_v:.6f} ({(1-pid_v/mlp_v)*100:.1f}% better) vs LSTM {lstm_v:.6f} ({(1-pid_v/lstm_v)*100:.1f}% better)")
    elif pid_v < mlp_v:
        print(f"✅ PID-Net beats MLP, LSTM slightly ahead")
    else:
        print(f"⚠️ PID: {pid_v:.6f} | MLP: {mlp_v:.6f} | LSTM: {lstm_v:.6f}")


if __name__ == "__main__":
    main()
