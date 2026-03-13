"""
PID-Attention: Temporally-aware attention mechanism.

Q/K/V projections use PID decomposition so queries carry:
- What IS relevant (P)
- What HAS BEEN relevant (I - accumulated context)  
- What's BECOMING relevant (D - rate of change)

Optimized for Apple Silicon (MPS) + 8GB RAM.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class PIDProjection(nn.Module):
    """
    PID-aware projection for Q/K/V.
    Replaces standard nn.Linear with P+I+D weighted combination.
    Uses vectorized ops (no python loops).
    """
    
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        
        # PID weight matrices
        self.W_p = nn.Linear(d_in, d_out, bias=False)
        self.W_i = nn.Linear(d_in, d_out, bias=False)
        self.W_d = nn.Linear(d_in, d_out, bias=False)
        
        # Gate: learns P/I/D mixture
        self.gate = nn.Linear(d_in, 3, bias=True)
        
        # Smaller init for I and D
        self.W_i.weight.data *= 0.1
        self.W_d.weight.data *= 0.1
        self.gate.bias.data = torch.tensor([1.0, 0.0, 0.0])  # start P-heavy
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, d_in)
        returns: (B, T, d_out)
        """
        B, T, D = x.shape
        
        # Integral: cumulative mean (vectorized)
        integral = torch.cumsum(x, dim=1) / torch.arange(1, T+1, device=x.device, dtype=x.dtype).view(1, T, 1)
        
        # Derivative: finite difference
        derivative = torch.zeros_like(x)
        derivative[:, 1:] = x[:, 1:] - x[:, :-1]
        
        # Gate
        g = F.softmax(self.gate(x), dim=-1)  # (B, T, 3)
        
        # PID projections
        out = (g[:, :, 0:1] * self.W_p(x) + 
               g[:, :, 1:2] * self.W_i(integral) + 
               g[:, :, 2:3] * self.W_d(derivative))
        
        return out


class PIDMultiHeadAttention(nn.Module):
    """
    Multi-head attention with PID-aware Q/K/V projections.
    
    Each head's Q/K/V carries temporal context through PID decomposition.
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        # PID projections for Q, K, V
        self.q_proj = PIDProjection(d_model, d_model)
        self.k_proj = PIDProjection(d_model, d_model)
        self.v_proj = PIDProjection(d_model, d_model)
        
        # Output projection (standard linear is fine here)
        self.o_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
        self.scale = math.sqrt(self.d_head)
    
    def forward(
        self, x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
    ) -> torch.Tensor:
        """
        x: (B, T, d_model)
        mask: optional (T, T) attention mask
        returns: (B, T, d_model)
        """
        B, T, D = x.shape
        
        # PID-aware Q/K/V
        Q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K = self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        # Q/K/V: (B, n_heads, T, d_head)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, H, T, T)
        
        # Causal mask
        if is_causal:
            causal_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, V)  # (B, H, T, d_head)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        
        return self.o_proj(out)


class PIDFeedForward(nn.Module):
    """Feed-forward with PID-aware projections."""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = PIDProjection(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)  # output proj can be standard
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.dropout(self.w2(self.act(self.w1(x))))


class PIDTransformerBlock(nn.Module):
    """Single transformer block with PID attention + PID FFN."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.RMSNorm(d_model)
        self.attn = PIDMultiHeadAttention(d_model, n_heads, dropout)
        self.norm2 = nn.RMSNorm(d_model)
        self.ffn = PIDFeedForward(d_model, d_ff, dropout)
    
    def forward(self, x: torch.Tensor, is_causal: bool = True) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), is_causal=is_causal)
        x = x + self.ffn(self.norm2(x))
        return x


class PIDTransformer(nn.Module):
    """
    Full PID-Transformer for language modeling.
    
    Sized for M2 MacBook with 8GB RAM:
    - d_model=256, n_heads=4, n_layers=6, d_ff=512
    - ~10M params → fits comfortably
    """
    
    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 6,
        d_ff: int = 512,
        max_seq_len: int = 256,
        dropout: float = 0.1,
        tie_weights: bool = True,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Embeddings
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.emb_dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            PIDTransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Output
        self.norm = nn.RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Weight tying
        if tie_weights:
            self.lm_head.weight = self.tok_emb.weight
        
        # Init
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self, 
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        input_ids: (B, T) token indices
        targets: (B, T) target token indices (for loss computation)
        """
        B, T = input_ids.shape
        assert T <= self.max_seq_len, f"Sequence length {T} > max {self.max_seq_len}"
        
        # Embeddings
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        x = self.emb_dropout(x)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x, is_causal=True)
        
        x = self.norm(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
        
        return {"logits": logits, "loss": loss}
    
    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
    
    def get_gate_stats(self) -> dict:
        """Collect gate statistics from all PID projections."""
        stats = {}
        for name, module in self.named_modules():
            if isinstance(module, PIDProjection):
                # Run a dummy forward to populate gates
                stats[name] = {
                    "gate_bias": module.gate.bias.data.softmax(0).tolist()
                }
        return stats

    @torch.no_grad()
    def generate(
        self, 
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
    ) -> torch.Tensor:
        """Autoregressive generation."""
        for _ in range(max_new_tokens):
            # Crop to max seq len
            idx_cond = input_ids[:, -self.max_seq_len:]
            
            # Forward
            out = self(idx_cond)
            logits = out["logits"][:, -1, :] / temperature
            
            # Top-k filtering
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids


# === Standard Transformer Baseline ===

class StandardTransformerBlock(nn.Module):
    """Standard transformer block for comparison."""
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.norm1 = nn.RMSNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.RMSNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
    
    def forward(self, x, is_causal=True):
        T = x.size(1)
        mask = None
        if is_causal:
            mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        h = self.norm1(x)
        x = x + self.attn(h, h, h, attn_mask=mask, is_causal=is_causal)[0]
        x = x + self.ffn(self.norm2(x))
        return x


class StandardTransformer(nn.Module):
    """Standard transformer baseline — same size as PID-Transformer."""
    
    def __init__(self, vocab_size=32000, d_model=256, n_heads=4, n_layers=6, 
                 d_ff=512, max_seq_len=256, dropout=0.1, tie_weights=True):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.emb_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            StandardTransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.RMSNorm(d_model)
        self.lm_head = nn.Linear(vocab_size, d_model, bias=False)  # will be tied
        if tie_weights:
            self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
            self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids, targets=None):
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        x = self.emb_dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return {"logits": logits, "loss": loss}
    
    def count_params(self):
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    # Quick test
    print("Testing PID-Transformer...")
    model = PIDTransformer(vocab_size=1000, d_model=128, n_heads=4, n_layers=2, d_ff=256, max_seq_len=64)
    x = torch.randint(0, 1000, (2, 32))
    out = model(x, targets=x)
    print(f"  Params: {model.count_params():,}")
    print(f"  Logits: {out['logits'].shape}")
    print(f"  Loss: {out['loss'].item():.4f}")
    print(f"  Gate stats: {model.get_gate_stats()}")
    
    # Generate
    prompt = torch.randint(0, 1000, (1, 5))
    generated = model.generate(prompt, max_new_tokens=20)
    print(f"  Generated: {generated.shape}")
    
    print("\nTesting Standard Transformer baseline...")
    baseline = StandardTransformer(vocab_size=1000, d_model=128, n_heads=4, n_layers=2, d_ff=256, max_seq_len=64)
    out2 = baseline(x, targets=x)
    print(f"  Params: {baseline.count_params():,}")
    print(f"  Loss: {out2['loss'].item():.4f}")
    
    print("\n✅ Both models work!")
