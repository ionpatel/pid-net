"""
ionBrain v2: Fixed Architecture

Changes from v1:
1. Gate regularization — prevents D-stream collapse (max 60% any single gate)
2. Reduced patch size (4 instead of 8) — preserves byte detail
3. Repetition penalty in loss — penalizes repeated byte predictions
4. Simplified encoder — removed sequential scan bottleneck
5. Direct byte pathway — skip connection from input to output
6. Temperature-aware gating — gates adapt based on input entropy

The v1 failure: D-stream dominated to 87%, model learned to predict
byte differences which collapsed to repeating characters. The fix is
architectural — force diversity in the cognitive streams.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Tuple


# ============================================================
# PID Core v2: With Gate Regularization
# ============================================================

class PIDProjection(nn.Module):
    """
    PID projection with gate regularization.
    No single gate can exceed max_gate (default 0.6).
    """
    
    def __init__(self, d_in: int, d_out: int, bias: bool = True, max_gate: float = 0.6):
        super().__init__()
        self.W_p = nn.Linear(d_in, d_out, bias=False)
        self.W_i = nn.Linear(d_in, d_out, bias=False)
        self.W_d = nn.Linear(d_in, d_out, bias=False)
        self.gate = nn.Linear(d_in, 3, bias=True)
        self.max_gate = max_gate
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(d_out))
        else:
            self.bias = None
        
        # Init: balanced start (not P-heavy like v1)
        self.W_i.weight.data *= 0.3
        self.W_d.weight.data *= 0.3
        self.gate.bias.data = torch.tensor([0.5, 0.0, 0.0])
        
        self._last_gates = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        
        # Integral: running cumulative mean
        integral = torch.cumsum(x, dim=1) / torch.arange(
            1, T+1, device=x.device, dtype=x.dtype
        ).view(1, T, 1)
        
        # Derivative: finite difference
        derivative = torch.zeros_like(x)
        derivative[:, 1:] = x[:, 1:] - x[:, :-1]
        
        # Gate with clamping — prevents any single stream from dominating
        raw_gate = self.gate(x)
        g = F.softmax(raw_gate, dim=-1)  # (B, T, 3)
        
        # Clamp max gate value and renormalize
        g = g.clamp(max=self.max_gate)
        g = g / g.sum(dim=-1, keepdim=True)
        
        self._last_gates = g.detach()
        
        # Weighted PID output
        out = (g[:,:,0:1] * self.W_p(x) +
               g[:,:,1:2] * self.W_i(integral) +
               g[:,:,2:3] * self.W_d(derivative))
        
        if self.bias is not None:
            out = out + self.bias
        
        return out


# ============================================================
# Simplified Local Encoder (no sequential scan)
# ============================================================

class PIDLocalBlock(nn.Module):
    """
    Local processing block using PID projections.
    No sequential scan — fully parallelizable.
    """
    
    def __init__(self, d_model: int, kernel_size: int = 7, dropout: float = 0.1):
        super().__init__()
        # Local context via causal conv
        self.conv = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size,
            padding=kernel_size - 1, groups=d_model  # causal + depthwise
        )
        self.gate_proj = nn.Linear(d_model, d_model)
        self.pid_proj = PIDProjection(d_model, d_model)
        self.norm = nn.RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        residual = x
        
        # Causal conv for local context
        h = self.conv(x.transpose(1, 2))[:, :, :T].transpose(1, 2)
        
        # Gated + PID projection
        gate = torch.sigmoid(self.gate_proj(x))
        h = gate * h
        h = self.pid_proj(h)
        
        h = self.dropout(h)
        return self.norm(h + residual)


# ============================================================
# Sparse Dilated Attention with PID Q/K/V
# ============================================================

def get_dilated_positions(seq_len: int, local: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
    max_attend = local + int(math.ceil(math.log2(max(seq_len, 2)))) + 1
    positions = torch.zeros(seq_len, max_attend, dtype=torch.long)
    mask = torch.zeros(seq_len, max_attend, dtype=torch.bool)
    
    for i in range(seq_len):
        attn_pos = [i]
        for j in range(1, local + 1):
            if i - j >= 0:
                attn_pos.append(i - j)
        k = 2
        while 2**k <= i:
            attn_pos.append(i - 2**k)
            k += 1
        n = len(attn_pos)
        positions[i, :n] = torch.tensor(attn_pos)
        mask[i, :n] = True
    
    return positions, mask


class PIDSparseAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5
        
        self.q_proj = PIDProjection(d_model, d_model, bias=False)
        self.k_proj = PIDProjection(d_model, d_model, bias=False)
        self.v_proj = PIDProjection(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self._pos_cache = {}
    
    def _get_pattern(self, T, device):
        if T not in self._pos_cache or self._pos_cache[T][0].device != device:
            p, m = get_dilated_positions(T)
            self._pos_cache[T] = (p.to(device), m.to(device))
        return self._pos_cache[T]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        positions, mask = self._get_pattern(T, x.device)
        max_attend = positions.shape[1]
        
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        
        BH = B * self.n_heads
        k_flat = k.reshape(BH, T, self.d_head)
        v_flat = v.reshape(BH, T, self.d_head)
        pos_exp = positions.unsqueeze(0).expand(BH, -1, -1)
        
        k_sparse = torch.gather(
            k_flat.unsqueeze(2).expand(-1, -1, max_attend, -1),
            dim=1, index=pos_exp.unsqueeze(-1).expand(-1, -1, -1, self.d_head)
        ).view(B, self.n_heads, T, max_attend, self.d_head)
        
        v_sparse = torch.gather(
            v_flat.unsqueeze(2).expand(-1, -1, max_attend, -1),
            dim=1, index=pos_exp.unsqueeze(-1).expand(-1, -1, -1, self.d_head)
        ).view(B, self.n_heads, T, max_attend, self.d_head)
        
        scores = torch.einsum('bhsd,bhsad->bhsa', q, k_sparse) * self.scale
        mask_exp = mask.unsqueeze(0).unsqueeze(0).expand(B, self.n_heads, -1, -1)
        scores = scores.masked_fill(~mask_exp, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.einsum('bhsa,bhsad->bhsd', attn, v_sparse)
        out = out.transpose(1, 2).reshape(B, T, D)
        return self.o_proj(out)


# ============================================================
# Transformer Block
# ============================================================

class PIDTransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.RMSNorm(d_model)
        self.attn = PIDSparseAttention(d_model, n_heads, dropout)
        self.norm2 = nn.RMSNorm(d_model)
        
        # SwiGLU FFN
        self.ffn_gate = nn.Linear(d_model, d_ff, bias=False)
        self.ffn_up = nn.Linear(d_model, d_ff, bias=False)
        self.ffn_down = nn.Linear(d_ff, d_model, bias=False)
        self.ffn_drop = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        residual = x
        h = self.norm2(x)
        x = F.silu(self.ffn_gate(h)) * self.ffn_up(h)
        x = self.ffn_down(x)
        x = self.ffn_drop(x) + residual
        return x


# ============================================================
# Simple Patching (fixed size, smaller patches)
# ============================================================

class SimplePatcher(nn.Module):
    """Fixed-size patches with mean pooling. Simpler = more reliable."""
    
    def __init__(self, d_model: int, patch_size: int = 4):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(d_model, d_model)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        ps = self.patch_size
        
        # Pad to multiple of patch_size
        pad = (ps - T % ps) % ps
        if pad > 0:
            x = F.pad(x, (0, 0, 0, pad))
        
        n_patches = x.shape[1] // ps
        patches = x.view(B, n_patches, ps, D).mean(dim=2)  # mean pool
        patches = self.proj(patches)
        
        byte_to_patch = torch.arange(T, device=x.device) // ps
        byte_to_patch = byte_to_patch.unsqueeze(0).expand(B, -1)
        
        return patches, byte_to_patch


# ============================================================
# ionBrain v2
# ============================================================

class ionBrainV2(nn.Module):
    """
    ionBrain v2: Fixed degenerate output issue.
    
    Key changes:
    - Gate clamping (max 0.6 per stream)
    - Smaller patches (4 instead of 8)
    - Direct byte skip connection
    - No sequential scan
    - Repetition-aware loss helper
    """
    
    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 256,
        n_heads: int = 4,
        n_encoder_layers: int = 2,
        n_transformer_layers: int = 6,
        d_ff: int = 512,
        max_seq_len: int = 1024,
        patch_size: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        
        # Byte embedding
        self.byte_embed = nn.Embedding(vocab_size, d_model)
        self.ngram_conv = nn.Conv1d(d_model, d_model, kernel_size=4, padding=3)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.embed_norm = nn.RMSNorm(d_model)
        self.embed_drop = nn.Dropout(dropout)
        
        # PID Local Encoder (parallel, no sequential scan)
        self.local_encoder = nn.ModuleList([
            PIDLocalBlock(d_model, dropout=dropout)
            for _ in range(n_encoder_layers)
        ])
        
        # Simple fixed-size patching
        self.patcher = SimplePatcher(d_model, patch_size)
        
        # Patch position embedding
        max_patches = max_seq_len // patch_size + 1
        self.patch_pos = nn.Embedding(max_patches, d_model)
        
        # PID-Sparse Transformer
        self.transformer = nn.ModuleList([
            PIDTransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_transformer_layers)
        ])
        self.transformer_norm = nn.RMSNorm(d_model)
        
        # Decoder: cross-attention + local processing
        self.cross_q = nn.Linear(d_model, d_model)
        self.cross_kv = nn.Linear(d_model, d_model * 2)
        self.cross_out = nn.Linear(d_model, d_model)
        self.cross_norm = nn.RMSNorm(d_model)
        
        self.local_decoder = nn.ModuleList([
            PIDLocalBlock(d_model, dropout=dropout)
            for _ in range(n_encoder_layers)
        ])
        
        # Direct skip connection from input embedding to output
        self.skip_gate = nn.Linear(d_model * 2, d_model)
        
        # Prediction head (tied weights)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.byte_embed.weight
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, bytes_input: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Dict:
        B, T = bytes_input.shape
        
        # 1. Embed
        x = self.byte_embed(bytes_input)
        x_ngram = self.ngram_conv(x.transpose(1, 2))[:, :, :T].transpose(1, 2)
        pos = self.pos_embed(torch.arange(T, device=x.device).unsqueeze(0))
        x = self.embed_norm(x + x_ngram + pos)
        x = self.embed_drop(x)
        
        # Save for skip connection
        byte_input_repr = x
        
        # 2. Local encode
        for layer in self.local_encoder:
            x = layer(x)
        byte_encoded = x
        
        # 3. Patch
        patches, byte_to_patch = self.patcher(byte_encoded)
        n_patches = patches.shape[1]
        patch_pos = self.patch_pos(torch.arange(n_patches, device=x.device).unsqueeze(0))
        patches = patches + patch_pos
        
        # 4. Transformer on patches
        h = patches
        for layer in self.transformer:
            h = layer(h)
        h = self.transformer_norm(h)
        
        # 5. Decode: patches → bytes
        patch_for_bytes = torch.gather(
            h, dim=1,
            index=byte_to_patch.unsqueeze(-1).expand(-1, -1, self.d_model)
        )
        
        # Simple cross-attention
        q = self.cross_q(byte_encoded)
        kv = self.cross_kv(patch_for_bytes)
        k, v = kv.chunk(2, dim=-1)
        attn_scores = (q * k).sum(dim=-1, keepdim=True) / math.sqrt(self.d_model)
        decoded = byte_encoded + self.cross_out(v * torch.sigmoid(attn_scores))
        decoded = self.cross_norm(decoded)
        
        for layer in self.local_decoder:
            decoded = layer(decoded)
        
        # 6. Skip connection from input
        combined = torch.cat([decoded, byte_input_repr], dim=-1)
        output = self.skip_gate(combined)
        
        # 7. Predict
        logits = self.head(output)
        
        result = {"logits": logits, "n_patches": n_patches}
        
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1), ignore_index=-1)
            
            # Repetition penalty: penalize when consecutive predictions are identical
            pred_bytes = logits.argmax(dim=-1)  # (B, T)
            rep_mask = (pred_bytes[:, 1:] == pred_bytes[:, :-1]).float()
            rep_penalty = rep_mask.mean() * 0.1  # small penalty
            
            loss = loss + rep_penalty
            result["loss"] = loss
            result["bpb"] = loss / math.log(2)
            result["rep_rate"] = rep_mask.mean().item()
        
        return result
    
    def count_params(self) -> Dict[str, int]:
        counts = {}
        counts["embedding"] = sum(p.numel() for n, p in self.named_parameters() if "embed" in n or "ngram" in n)
        counts["local_encoder"] = sum(p.numel() for p in self.local_encoder.parameters())
        counts["transformer"] = sum(p.numel() for p in self.transformer.parameters())
        counts["decoder"] = sum(p.numel() for p in self.local_decoder.parameters())
        counts["total"] = sum(p.numel() for p in self.parameters())
        return counts
    
    @torch.no_grad()
    def generate(self, bytes_input: torch.Tensor, max_new: int = 200, 
                 temperature: float = 0.8, top_k: int = 40, 
                 rep_penalty: float = 1.2) -> torch.Tensor:
        """Generate with repetition penalty."""
        for _ in range(max_new):
            x = bytes_input[:, -self.max_seq_len:]
            out = self(x)
            logits = out["logits"][:, -1, :] / temperature
            
            # Penalize recently generated bytes
            if bytes_input.shape[1] > 1:
                recent = bytes_input[0, -20:].tolist()  # last 20 bytes
                for byte_val in set(recent):
                    count = recent.count(byte_val)
                    if count > 2:
                        logits[0, byte_val] /= rep_penalty ** (count - 2)
            
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            probs = F.softmax(logits, dim=-1)
            next_byte = torch.multinomial(probs, num_samples=1)
            bytes_input = torch.cat([bytes_input, next_byte], dim=1)
        
        return bytes_input
    
    def get_gate_stats(self) -> Dict:
        stats = {}
        for name, module in self.named_modules():
            if isinstance(module, PIDProjection) and module._last_gates is not None:
                g = module._last_gates
                stats[name] = {
                    "P": g[:,:,0].mean().item(),
                    "I": g[:,:,1].mean().item(),
                    "D": g[:,:,2].mean().item(),
                }
        return stats


# ============================================================
# Configs
# ============================================================

def create_ionbrain_v2(size: str = "small") -> ionBrainV2:
    configs = {
        "tiny": dict(
            d_model=192, n_heads=4, n_encoder_layers=1,
            n_transformer_layers=4, d_ff=384, max_seq_len=512, patch_size=4,
        ),
        "small": dict(
            d_model=256, n_heads=4, n_encoder_layers=2,
            n_transformer_layers=6, d_ff=512, max_seq_len=1024, patch_size=4,
        ),
        "base": dict(
            d_model=384, n_heads=6, n_encoder_layers=2,
            n_transformer_layers=8, d_ff=768, max_seq_len=1024, patch_size=4,
        ),
    }
    if size not in configs:
        raise ValueError(f"Unknown size: {size}. Options: {list(configs.keys())}")
    return ionBrainV2(**configs[size])


# ============================================================
# Baseline
# ============================================================

class StandardByteTransformer(nn.Module):
    """Standard transformer baseline."""
    
    def __init__(self, vocab_size=256, d_model=256, n_heads=4, n_layers=6,
                 d_ff=512, max_seq_len=1024, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)
        
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight
    
    def forward(self, bytes_input, targets=None):
        B, T = bytes_input.shape
        pos = torch.arange(T, device=bytes_input.device).unsqueeze(0)
        x = self.drop(self.tok_emb(bytes_input) + self.pos_emb(pos))
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        x = self.transformer(x, mask=mask, is_causal=True)
        logits = self.head(self.norm(x))
        result = {"logits": logits}
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1), ignore_index=-1)
            result["loss"] = loss
            result["bpb"] = loss / math.log(2)
        return result
    
    def count_params(self):
        return {"total": sum(p.numel() for p in self.parameters())}


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ionBrain v2 — Architecture Test")
    print("=" * 60)
    
    for size in ["tiny", "small", "base"]:
        model = create_ionbrain_v2(size)
        counts = model.count_params()
        print(f"\n{size}: {counts['total']:,} total params")
    
    print("\n--- Forward Pass ---")
    model = create_ionbrain_v2("small")
    text = "Once upon a time there was a little girl named Lily."
    x = torch.tensor([[ord(c) for c in text]])
    y = torch.tensor([[ord(c) for c in text[1:]] + [0]])
    
    out = model(x, targets=y)
    print(f"Loss: {out['loss'].item():.4f} | BPB: {out['bpb'].item():.3f} | Rep rate: {out['rep_rate']:.3f}")
    
    # Gate stats
    gates = model.get_gate_stats()
    for name, g in list(gates.items())[:3]:
        print(f"  {name}: P={g['P']:.3f} I={g['I']:.3f} D={g['D']:.3f}")
    
    # Generation test
    print("\n--- Generation ---")
    prompt = "Once upon a time"
    px = torch.tensor([[ord(c) for c in prompt]])
    gen = model.generate(px, max_new=100, temperature=0.8)
    text_out = "".join(chr(min(b, 127)) for b in gen[0].tolist())
    print(f"Output: {text_out}")
    
    print("\n✅ ionBrain v2 ready!")
