"""
PID-Net: Control-Theoretic Neural Layer

A drop-in replacement for nn.Linear that decomposes weights into
Proportional (current), Integral (accumulated), and Derivative (rate of change)
components with learned gating for adaptive computation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class PIDLinear(nn.Module):
    """
    PID Linear Layer — replaces nn.Linear with PID-structured weights.
    
    For input sequence x of shape (B, T, d_in):
      P-term: W_p @ x[t]                    (react to current)
      I-term: W_i @ ema(x, α)[t]            (react to accumulated history)
      D-term: W_d @ (x[t] - x[t-1])         (react to change)
      
    Gate: g = softmax(W_gate @ [x; ema; dx])  (learn which to use)
    Output: g_p * P + g_i * I + g_d * D + bias
    """
    
    def __init__(
        self,
        d_in: int,
        d_out: int,
        bias: bool = True,
        init_alpha: float = 0.9,      # integral decay initialization
        init_beta: float = 0.5,        # derivative smoothing initialization
        gate_type: str = "softmax",    # softmax | sigmoid | straight_through
        sparsity_threshold: float = 0.01,  # skip component if gate < this
        adaptive_compute: bool = True,  # enable dynamic component skipping
    ):
        super().__init__()
        
        self.d_in = d_in
        self.d_out = d_out
        self.gate_type = gate_type
        self.sparsity_threshold = sparsity_threshold
        self.adaptive_compute = adaptive_compute
        
        # === PID Weight Matrices ===
        self.W_p = nn.Parameter(torch.empty(d_out, d_in))  # Proportional
        self.W_i = nn.Parameter(torch.empty(d_out, d_in))  # Integral
        self.W_d = nn.Parameter(torch.empty(d_out, d_in))  # Derivative
        
        # === Temporal Parameters (per-channel) ===
        # α: integral decay — how much history to keep (sigmoid → (0,1))
        self._alpha_logit = nn.Parameter(torch.full((d_in,), self._inv_sigmoid(init_alpha)))
        # β: derivative smoothing (sigmoid → (0,1))
        self._beta_logit = nn.Parameter(torch.full((d_in,), self._inv_sigmoid(init_beta)))
        
        # === Gating Network ===
        # Input: concatenation of [x, integral, derivative] → 3 gate values
        self.W_gate = nn.Linear(3 * d_in, 3, bias=True)
        
        # === Bias ===
        if bias:
            self.bias = nn.Parameter(torch.zeros(d_out))
        else:
            self.register_parameter('bias', None)
        
        # === State for streaming / recurrent use ===
        self.register_buffer('_integral_state', None)
        self.register_buffer('_prev_input', None)
        self.register_buffer('_deriv_state', None)
        
        # === Diagnostics ===
        self.register_buffer('_gate_history', None)  # track gate activations
        
        self._init_weights()
    
    def _init_weights(self):
        """Kaiming init for P, smaller init for I and D."""
        # P gets standard init — it's the primary signal
        nn.init.kaiming_uniform_(self.W_p, a=math.sqrt(5))
        
        # I and D start smaller — let the model learn to use them
        nn.init.kaiming_uniform_(self.W_i, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_d, a=math.sqrt(5))
        self.W_i.data *= 0.1
        self.W_d.data *= 0.1
        
        # Gate bias: initially favor proportional
        nn.init.constant_(self.W_gate.bias, 0)
        self.W_gate.bias.data[0] = 1.0  # bias toward P initially
    
    @staticmethod
    def _inv_sigmoid(x: float) -> float:
        """Inverse sigmoid for initialization."""
        return math.log(x / (1 - x + 1e-8))
    
    @property
    def alpha(self) -> torch.Tensor:
        """Integral decay factor ∈ (0, 1)."""
        return torch.sigmoid(self._alpha_logit)
    
    @property
    def beta(self) -> torch.Tensor:
        """Derivative smoothing factor ∈ (0, 1)."""
        return torch.sigmoid(self._beta_logit)
    
    def _compute_integral(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute exponential moving average (soft integral).
        
        S(t) = α * S(t-1) + (1-α) * x(t)
        
        Args:
            x: (B, T, d_in)
        Returns:
            integral: (B, T, d_in)
        """
        B, T, D = x.shape
        alpha = self.alpha.unsqueeze(0).unsqueeze(0)  # (1, 1, d_in)
        
        integral = torch.zeros_like(x)
        
        # Initialize from state if available (streaming mode)
        if self._integral_state is not None and self._integral_state.shape[0] == B:
            s = self._integral_state
        else:
            s = torch.zeros(B, D, device=x.device, dtype=x.dtype)
        
        # Sequential scan (can be parallelized with associative scan)
        for t in range(T):
            s = alpha.squeeze(1) * s + (1 - alpha.squeeze(1)) * x[:, t, :]
            integral[:, t, :] = s
        
        # Save state for streaming
        self._integral_state = s.detach()
        
        return integral
    
    def _compute_derivative(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute smoothed finite difference (derivative approximation).
        
        Δ(t) = β * Δ(t-1) + (1-β) * (x(t) - x(t-1))
        
        Args:
            x: (B, T, d_in)
        Returns:
            derivative: (B, T, d_in)
        """
        B, T, D = x.shape
        beta = self.beta.unsqueeze(0).unsqueeze(0)
        
        # Raw differences
        if self._prev_input is not None and self._prev_input.shape[0] == B:
            x_prev = torch.cat([self._prev_input.unsqueeze(1), x[:, :-1, :]], dim=1)
        else:
            x_prev = torch.cat([torch.zeros(B, 1, D, device=x.device, dtype=x.dtype), x[:, :-1, :]], dim=1)
        
        raw_diff = x - x_prev  # (B, T, D)
        
        # Smoothed derivative
        derivative = torch.zeros_like(x)
        if self._deriv_state is not None and self._deriv_state.shape[0] == B:
            d = self._deriv_state
        else:
            d = torch.zeros(B, D, device=x.device, dtype=x.dtype)
        
        for t in range(T):
            d = beta.squeeze(1) * d + (1 - beta.squeeze(1)) * raw_diff[:, t, :]
            derivative[:, t, :] = d
        
        # Save state
        self._prev_input = x[:, -1, :].detach()
        self._deriv_state = d.detach()
        
        return derivative
    
    def _compute_gates(
        self, x: torch.Tensor, integral: torch.Tensor, derivative: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute adaptive gates that decide P/I/D mixture.
        
        Args:
            x, integral, derivative: all (B, T, d_in)
        Returns:
            gates: (B, T, 3) — weights for [P, I, D]
        """
        gate_input = torch.cat([x, integral, derivative], dim=-1)  # (B, T, 3*d_in)
        gate_logits = self.W_gate(gate_input)  # (B, T, 3)
        
        if self.gate_type == "softmax":
            gates = F.softmax(gate_logits, dim=-1)
        elif self.gate_type == "sigmoid":
            gates = torch.sigmoid(gate_logits)
        else:
            # Straight-through estimator for hard gating
            gates = F.softmax(gate_logits, dim=-1)
            hard = (gates > 0.33).float()
            gates = hard + gates - gates.detach()  # STE
        
        return gates
    
    def forward(
        self, x: torch.Tensor, return_gates: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: (B, T, d_in) or (B, d_in) — if 2D, treated as T=1
            return_gates: if True, also return gate values for analysis
            
        Returns:
            y: (B, T, d_out) or (B, d_out)
            gates: (B, T, 3) — only if return_gates=True
        """
        squeeze = False
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, d_in)
            squeeze = True
        
        B, T, D = x.shape
        
        # === Compute temporal signals ===
        integral = self._compute_integral(x)      # (B, T, d_in)
        derivative = self._compute_derivative(x)   # (B, T, d_in)
        
        # === Compute gates ===
        gates = self._compute_gates(x, integral, derivative)  # (B, T, 3)
        g_p = gates[:, :, 0:1]  # (B, T, 1)
        g_i = gates[:, :, 1:2]
        g_d = gates[:, :, 2:3]
        
        # === PID computation with optional sparsity ===
        y = torch.zeros(B, T, self.d_out, device=x.device, dtype=x.dtype)
        
        if self.adaptive_compute and not self.training:
            # During inference, skip components with low gate values
            mean_gp = g_p.mean().item()
            mean_gi = g_i.mean().item()
            mean_gd = g_d.mean().item()
            
            if mean_gp > self.sparsity_threshold:
                y = y + g_p * F.linear(x, self.W_p)
            if mean_gi > self.sparsity_threshold:
                y = y + g_i * F.linear(integral, self.W_i)
            if mean_gd > self.sparsity_threshold:
                y = y + g_d * F.linear(derivative, self.W_d)
        else:
            # During training, always compute all components
            p_out = F.linear(x, self.W_p)           # (B, T, d_out)
            i_out = F.linear(integral, self.W_i)     # (B, T, d_out)
            d_out = F.linear(derivative, self.W_d)   # (B, T, d_out)
            
            y = g_p * p_out + g_i * i_out + g_d * d_out
        
        if self.bias is not None:
            y = y + self.bias
        
        # Store gate history for diagnostics
        if return_gates or self.training:
            self._gate_history = gates.detach()
        
        if squeeze:
            y = y.squeeze(1)
            if return_gates:
                gates = gates.squeeze(1)
        
        if return_gates:
            return y, gates
        return y
    
    def reset_state(self):
        """Reset temporal state (call between sequences)."""
        self._integral_state = None
        self._prev_input = None
        self._deriv_state = None
    
    def get_gate_stats(self) -> dict:
        """Return gate activation statistics for analysis."""
        if self._gate_history is None:
            return {}
        g = self._gate_history
        return {
            "gate_p_mean": g[:, :, 0].mean().item(),
            "gate_i_mean": g[:, :, 1].mean().item(),
            "gate_d_mean": g[:, :, 2].mean().item(),
            "gate_p_std": g[:, :, 0].std().item(),
            "gate_i_std": g[:, :, 1].std().item(),
            "gate_d_std": g[:, :, 2].std().item(),
        }
    
    def extra_repr(self) -> str:
        return (
            f"d_in={self.d_in}, d_out={self.d_out}, "
            f"gate={self.gate_type}, adaptive={self.adaptive_compute}"
        )


class PIDBlock(nn.Module):
    """
    A full PID block with normalization and residual connection.
    
    x → LayerNorm → PIDLinear → GELU → PIDLinear → + x
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        **pid_kwargs,
    ):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        
        self.norm = nn.LayerNorm(d_model)
        self.pid1 = PIDLinear(d_model, d_ff, **pid_kwargs)
        self.pid2 = PIDLinear(d_ff, d_model, **pid_kwargs)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.pid1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.pid2(x)
        x = self.dropout(x)
        return x + residual
    
    def reset_state(self):
        self.pid1.reset_state()
        self.pid2.reset_state()


class PIDNet(nn.Module):
    """
    Full PID-Net model for sequence modeling.
    
    Stack of PID blocks with optional embedding layer.
    """
    
    def __init__(
        self,
        d_input: int,
        d_model: int,
        d_output: int,
        n_layers: int = 4,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        **pid_kwargs,
    ):
        super().__init__()
        
        self.input_proj = nn.Linear(d_input, d_model)
        self.blocks = nn.ModuleList([
            PIDBlock(d_model, d_ff, dropout, **pid_kwargs)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, d_output)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_input)
        Returns:
            y: (B, T, d_output)
        """
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = self.output_proj(x)
        return x
    
    def reset_state(self):
        for block in self.blocks:
            block.reset_state()
    
    def get_all_gate_stats(self) -> dict:
        """Collect gate stats from all PID layers."""
        stats = {}
        for i, block in enumerate(self.blocks):
            stats[f"block_{i}_ff1"] = block.pid1.get_gate_stats()
            stats[f"block_{i}_ff2"] = block.pid2.get_gate_stats()
        return stats
    
    def count_active_params(self) -> Tuple[int, int]:
        """Count total vs active parameters (based on gate sparsity)."""
        total = sum(p.numel() for p in self.parameters())
        # Could be extended to estimate active params based on gates
        return total, total


# === Utility: Sparsity Loss ===

def pid_sparsity_loss(model: nn.Module, target_entropy: float = 0.5) -> torch.Tensor:
    """
    Regularization loss that encourages decisive gate activations.
    
    Low entropy = model is confident about which PID components to use.
    This leads to sparser, faster inference.
    """
    loss = torch.tensor(0.0, device=next(model.parameters()).device)
    count = 0
    
    for module in model.modules():
        if isinstance(module, PIDLinear) and module._gate_history is not None:
            gates = module._gate_history  # (B, T, 3)
            # Entropy of gate distribution
            entropy = -(gates * (gates + 1e-8).log()).sum(dim=-1).mean()
            loss = loss + (entropy - target_entropy).abs()
            count += 1
    
    return loss / max(count, 1)


if __name__ == "__main__":
    # === Quick Test ===
    torch.manual_seed(42)
    
    B, T, D_in, D_out = 4, 32, 64, 64
    
    # Test PIDLinear
    layer = PIDLinear(D_in, D_out)
    x = torch.randn(B, T, D_in)
    y, gates = layer(x, return_gates=True)
    
    print(f"Input:  {x.shape}")
    print(f"Output: {y.shape}")
    print(f"Gates:  {gates.shape}")
    print(f"Gate stats: {layer.get_gate_stats()}")
    print()
    
    # Test PIDNet
    model = PIDNet(d_input=16, d_model=64, d_output=1, n_layers=3)
    x = torch.randn(B, T, 16)
    y = model(x)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"PIDNet output: {y.shape}")
    print(f"Total params: {total_params:,}")
    print(f"Gate stats: {model.get_all_gate_stats()}")
    
    # Backward pass test
    loss = y.mean()
    loss.backward()
    print("\nBackward pass: ✅")
    
    # Sparsity loss
    s_loss = pid_sparsity_loss(model)
    print(f"Sparsity loss: {s_loss.item():.4f}")
