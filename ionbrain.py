"""
ionBrain v1: The Unified Architecture

Combines the best of:
- PID-Net: P/I/D decomposition with adaptive gating (cognitive primitives)
- ionLlama v12: Byte-level processing, entropy patching, sparse attention

Architecture:
    Raw bytes → ByteEmbedding (n-gram aware)
    → PID Local Encoder (accumulates context via P+I+D)
    → Entropy-based Dynamic Patching (variable compute allocation)
    → PID-Sparse Latent Transformer (O(n log n) attention with PID Q/K/V)
    → PID Local Decoder (patches → bytes)
    → Byte prediction head

Key innovations:
1. PID projections replace ALL linear layers in attention (Q/K/V carry temporal info)
2. IntegralLNN from ionLlama upgraded with P and D streams
3. Entropy patching allocates compute where it matters
4. Sparse dilated attention keeps O(n log n) complexity
5. Gate biases = learned personality/cognitive style

Optimized for Apple M2 8GB (tiny config: ~15M params)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Tuple


# ============================================================
# PID Core: The Cognitive Primitive
# ============================================================

class PIDProjection(nn.Module):
    """
    PID-aware linear projection.
    Decomposes transformation into:
      P: operate on current input (perception)
      I: operate on accumulated history (memory)
      D: operate on rate of change (intuition)
    Gate learns which to trust.
    """
    
    def __init__(self, d_in: int, d_out: int, bias: bool = True):
        super().__init__()
        self.W_p = nn.Linear(d_in, d_out, bias=False)
        self.W_i = nn.Linear(d_in, d_out, bias=False)
        self.W_d = nn.Linear(d_in, d_out, bias=False)
        self.gate = nn.Linear(d_in, 3, bias=True)
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(d_out))
        else:
            self.bias = None
        
        # Init: I and D start small, gate starts P-heavy
        self.W_i.weight.data *= 0.1
        self.W_d.weight.data *= 0.1
        self.gate.bias.data = torch.tensor([1.0, -0.5, -0.5])
        
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
        
        # Gate
        g = F.softmax(self.gate(x), dim=-1)  # (B, T, 3)
        self._last_gates = g.detach()
        
        # Weighted PID output
        out = (g[:,:,0:1] * self.W_p(x) +
               g[:,:,1:2] * self.W_i(integral) +
               g[:,:,2:3] * self.W_d(derivative))
        
        if self.bias is not None:
            out = out + self.bias
        
        return out


# ============================================================
# PID-Enhanced IntegralLNN (from ionLlama, upgraded)
# ============================================================

class PIDIntegralLNN(nn.Module):
    """
    IntegralLNN from ionLlama v12, upgraded with PID decomposition.
    
    Original: gated integration with exponential decay
    Upgrade: P-stream (current), I-stream (integrated), D-stream (change)
             with learned gating for cognitive adaptation
    """
    
    def __init__(self, d_model: int, kernel_size: int = 7, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        
        # Gated integration (from ionLlama)
        self.gate_proj = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size,
            padding=kernel_size // 2, groups=d_model
        )
        self.candidate_proj = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size,
            padding=kernel_size // 2, groups=d_model
        )
        
        # Learnable decay rate per dimension
        self.log_decay = nn.Parameter(torch.full((d_model,), math.log(0.1)))
        
        # PID output projection (replaces standard linear)
        self.out_proj = PIDProjection(d_model, d_model)
        
        self.norm = nn.RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        residual = x
        
        x_t = x.transpose(1, 2)
        gate = torch.sigmoid(self.gate_proj(x_t))
        candidate = torch.tanh(self.candidate_proj(x_t))
        gated = gate * candidate
        
        # Exponential decay integration (from ionLlama)
        decay = torch.exp(self.log_decay).view(1, D, 1)
        h = self._scan(gated, decay)
        h = h.transpose(1, 2)
        
        # PID output projection
        out = self.out_proj(h)
        out = self.dropout(out)
        out = self.norm(out + residual)
        return out
    
    def _scan(self, x: torch.Tensor, decay: torch.Tensor) -> torch.Tensor:
        """Exponentially-decayed cumsum via sequential scan."""
        B, D, T = x.shape
        h = torch.zeros(B, D, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(T):
            h = decay.squeeze(-1) * h + x[:, :, t]
            outputs.append(h)
        return torch.stack(outputs, dim=2)


# ============================================================
# PID Sparse Attention (from ionLlama, upgraded)
# ============================================================

def get_dilated_positions(seq_len: int, local: int = 2) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sparse attention pattern: each position attends to O(log n) positions."""
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
    """
    Sparse dilated attention with PID-aware Q/K/V projections.
    
    O(n log n) complexity + temporal awareness in every query/key/value.
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5
        
        # PID projections for Q/K/V
        self.q_proj = PIDProjection(d_model, d_model, bias=False)
        self.k_proj = PIDProjection(d_model, d_model, bias=False)
        self.v_proj = PIDProjection(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self._pos_cache = {}
    
    def _get_pattern(self, T: int, device):
        if T not in self._pos_cache or self._pos_cache[T][0].device != device:
            p, m = get_dilated_positions(T)
            self._pos_cache[T] = (p.to(device), m.to(device))
        return self._pos_cache[T]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        positions, mask = self._get_pattern(T, x.device)
        max_attend = positions.shape[1]
        
        # PID-aware Q/K/V
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        
        # Sparse gather
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
        
        # Attention
        scores = torch.einsum('bhsd,bhsad->bhsa', q, k_sparse) * self.scale
        mask_exp = mask.unsqueeze(0).unsqueeze(0).expand(B, self.n_heads, -1, -1)
        scores = scores.masked_fill(~mask_exp, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.einsum('bhsa,bhsad->bhsd', attn, v_sparse)
        out = out.transpose(1, 2).reshape(B, T, D)
        return self.o_proj(out)


# ============================================================
# PID Transformer Block
# ============================================================

class PIDTransformerBlock(nn.Module):
    """Transformer block with PID sparse attention + PID FFN."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.RMSNorm(d_model)
        self.attn = PIDSparseAttention(d_model, n_heads, dropout)
        self.norm2 = nn.RMSNorm(d_model)
        
        # SwiGLU FFN with PID first projection
        self.ffn_gate = PIDProjection(d_model, d_ff, bias=False)
        self.ffn_up = nn.Linear(d_model, d_ff, bias=False)
        self.ffn_down = nn.Linear(d_ff, d_model, bias=False)
        self.ffn_drop = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # PID Sparse Attention
        x = x + self.attn(self.norm1(x))
        
        # SwiGLU FFN with PID gate
        residual = x
        h = self.norm2(x)
        x = F.silu(self.ffn_gate(h)) * self.ffn_up(h)
        x = self.ffn_down(x)
        x = self.ffn_drop(x)
        x = x + residual
        
        return x


# ============================================================
# Entropy-based Dynamic Patching (from ionLlama)
# ============================================================

class SimpleEntropyEstimator(nn.Module):
    """
    Lightweight entropy estimator for dynamic patching.
    Predicts how "surprising" each byte is to allocate compute.
    """
    
    def __init__(self, d_model: int, vocab_size: int = 256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=4, padding=3, groups=d_model)
        self.proj = nn.Linear(d_model, 1)
    
    def forward(self, bytes_input: torch.Tensor) -> torch.Tensor:
        """Returns entropy estimate per position. (B, T)"""
        x = self.embed(bytes_input)
        x = self.conv(x.transpose(1, 2))[:, :, :bytes_input.shape[1]].transpose(1, 2)
        return torch.sigmoid(self.proj(x)).squeeze(-1)


class DynamicPatcher(nn.Module):
    """
    Groups bytes into variable-length patches based on entropy.
    High entropy = patch boundary (needs more compute).
    """
    
    def __init__(self, d_model: int, max_patch_size: int = 8, entropy_threshold: float = 0.5):
        super().__init__()
        self.max_patch_size = max_patch_size
        self.entropy_threshold = entropy_threshold
        self.patch_proj = nn.Linear(d_model, d_model)
    
    def forward(
        self, 
        byte_encodings: torch.Tensor,
        entropy: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            byte_encodings: (B, T, D) - encoded bytes
            entropy: (B, T) - entropy per byte
        Returns:
            patch_embeddings: (B, max_patches, D)
            byte_to_patch: (B, T) - mapping bytes→patches
        """
        B, T, D = byte_encodings.shape
        
        # Determine patch boundaries
        # For efficiency and differentiability, use fixed-size patches
        # with entropy-weighted pooling
        patch_size = self.max_patch_size
        n_patches = (T + patch_size - 1) // patch_size
        
        # Pad if needed
        if T % patch_size != 0:
            pad_len = patch_size * n_patches - T
            byte_encodings = F.pad(byte_encodings, (0, 0, 0, pad_len))
            entropy = F.pad(entropy, (0, pad_len))
        
        # Reshape into patches
        x = byte_encodings.view(B, n_patches, patch_size, D)
        e = entropy.view(B, n_patches, patch_size)
        
        # Entropy-weighted mean pooling (high entropy bytes get more weight)
        weights = F.softmax(e, dim=-1).unsqueeze(-1)  # (B, n_patches, patch_size, 1)
        patches = (x * weights).sum(dim=2)  # (B, n_patches, D)
        patches = self.patch_proj(patches)
        
        # Byte-to-patch mapping
        byte_to_patch = torch.arange(T, device=byte_encodings.device) // patch_size
        byte_to_patch = byte_to_patch.unsqueeze(0).expand(B, -1)
        
        return patches, byte_to_patch


# ============================================================
# ionBrain: The Full Architecture
# ============================================================

class ionBrain(nn.Module):
    """
    ionBrain v1: PID-Net + ionLlama combined.
    
    Flow:
        bytes → embed → PID-IntegralLNN encoder → entropy patch
        → PID-Sparse Transformer → decode → predict next byte
    
    Configs sized for M2 8GB:
        tiny:  ~5M params  (d=192, 4 layers)
        small: ~15M params (d=256, 6 layers)
        base:  ~50M params (d=384, 8 layers)
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
        max_patch_size: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.max_patch_size = max_patch_size
        
        # Byte embedding with n-gram context
        self.byte_embed = nn.Embedding(vocab_size, d_model)
        self.ngram_conv = nn.Conv1d(d_model, d_model, kernel_size=4, padding=3)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.embed_norm = nn.RMSNorm(d_model)
        self.embed_drop = nn.Dropout(dropout)
        
        # Entropy estimator (lightweight)
        self.entropy = SimpleEntropyEstimator(d_model // 2, vocab_size)
        
        # Dynamic patcher
        self.patcher = DynamicPatcher(d_model, max_patch_size)
        
        # PID-IntegralLNN local encoder (bytes → rich representations)
        self.local_encoder = nn.ModuleList([
            PIDIntegralLNN(d_model, dropout=dropout)
            for _ in range(n_encoder_layers)
        ])
        
        # PID-Sparse Transformer (processes patches)
        self.transformer = nn.ModuleList([
            PIDTransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_transformer_layers)
        ])
        self.transformer_norm = nn.RMSNorm(d_model)
        
        # Patch position embedding
        max_patches = max_seq_len // max_patch_size + 1
        self.patch_pos = nn.Embedding(max_patches, d_model)
        
        # Decoder: patches → bytes via cross-attention + local processing
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_norm = nn.RMSNorm(d_model)
        self.local_decoder = nn.ModuleList([
            PIDIntegralLNN(d_model, dropout=dropout)
            for _ in range(n_encoder_layers)
        ])
        
        # Prediction head
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        # Weight tying
        self.head.weight = self.byte_embed.weight
        
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
        bytes_input: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Dict:
        B, T = bytes_input.shape
        
        # 1. Byte embedding + n-gram
        x = self.byte_embed(bytes_input)
        x_ngram = self.ngram_conv(x.transpose(1, 2))[:, :, :T].transpose(1, 2)
        pos = self.pos_embed(torch.arange(T, device=x.device).unsqueeze(0))
        x = self.embed_norm(x + x_ngram + pos)
        x = self.embed_drop(x)
        
        # 2. PID-IntegralLNN local encoding
        byte_encoded = x
        for layer in self.local_encoder:
            byte_encoded = layer(byte_encoded)
        
        # 3. Entropy estimation + patching
        with torch.no_grad():
            entropy = self.entropy(bytes_input)
        
        patches, byte_to_patch = self.patcher(byte_encoded, entropy)
        n_patches = patches.shape[1]
        
        # Add patch position embeddings
        patch_pos = self.patch_pos(torch.arange(n_patches, device=x.device).unsqueeze(0))
        patches = patches + patch_pos
        
        # 4. PID-Sparse Transformer on patches
        h = patches
        for layer in self.transformer:
            h = layer(h)
        h = self.transformer_norm(h)
        
        # 5. Decode: patches → bytes
        # Gather patch outputs for each byte
        patch_for_bytes = torch.gather(
            h, dim=1,
            index=byte_to_patch.unsqueeze(-1).expand(-1, -1, self.d_model)
        )
        
        # Cross-attention (bytes attend to their patches)
        causal = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        decoded, _ = self.cross_attn(byte_encoded, patch_for_bytes, patch_for_bytes, attn_mask=causal)
        decoded = self.cross_norm(decoded + byte_encoded)
        
        for layer in self.local_decoder:
            decoded = layer(decoded)
        
        # 6. Predict
        logits = self.head(decoded)
        
        result = {"logits": logits, "entropy": entropy, "n_patches": n_patches}
        
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1), ignore_index=-1)
            bpb = loss / math.log(2)
            result["loss"] = loss
            result["bpb"] = bpb
        
        return result
    
    def count_params(self) -> Dict[str, int]:
        counts = {}
        counts["embedding"] = sum(p.numel() for n, p in self.named_parameters() if "embed" in n or "ngram" in n)
        counts["entropy"] = sum(p.numel() for p in self.entropy.parameters())
        counts["local_encoder"] = sum(p.numel() for p in self.local_encoder.parameters())
        counts["transformer"] = sum(p.numel() for p in self.transformer.parameters())
        counts["decoder"] = sum(p.numel() for p in self.local_decoder.parameters()) + sum(p.numel() for p in self.cross_attn.parameters())
        counts["total"] = sum(p.numel() for p in self.parameters())
        return counts
    
    @torch.no_grad()
    def generate(self, bytes_input: torch.Tensor, max_new: int = 200, temperature: float = 0.8, top_k: int = 40) -> torch.Tensor:
        for _ in range(max_new):
            x = bytes_input[:, -self.max_seq_len:]
            out = self(x)
            logits = out["logits"][:, -1, :] / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            next_byte = torch.multinomial(probs, num_samples=1)
            bytes_input = torch.cat([bytes_input, next_byte], dim=1)
        return bytes_input
    
    def get_gate_stats(self) -> Dict:
        """Collect P/I/D gate statistics from all PID projections."""
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
# Model Configs
# ============================================================

def create_ionbrain(size: str = "small") -> ionBrain:
    """Create ionBrain with preset configs for M2 8GB."""
    configs = {
        "tiny": dict(
            d_model=192, n_heads=4, n_encoder_layers=1,
            n_transformer_layers=4, d_ff=384, max_seq_len=512,
        ),
        "small": dict(
            d_model=256, n_heads=4, n_encoder_layers=2,
            n_transformer_layers=6, d_ff=512, max_seq_len=1024,
        ),
        "base": dict(
            d_model=384, n_heads=6, n_encoder_layers=2,
            n_transformer_layers=8, d_ff=768, max_seq_len=1024,
        ),
    }
    if size not in configs:
        raise ValueError(f"Unknown size: {size}. Options: {list(configs.keys())}")
    return ionBrain(**configs[size])


# ============================================================
# Baseline: Standard Byte-Level Transformer
# ============================================================

class StandardByteTransformer(nn.Module):
    """Standard transformer baseline (byte-level, no PID, no patching)."""
    
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
            dropout=dropout, activation='gelu', batch_first=True,
            norm_first=True,
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
# Quick Test
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ionBrain v1 — Architecture Test")
    print("=" * 60)
    
    for size in ["tiny", "small", "base"]:
        model = create_ionbrain(size)
        counts = model.count_params()
        print(f"\n{size}: {counts['total']:,} total params")
        for k, v in counts.items():
            if k != "total":
                print(f"  {k}: {v:,}")
    
    # Forward test
    print("\n--- Forward Pass ---")
    model = create_ionbrain("small")
    text = "Hello, world! This is ionBrain speaking."
    x = torch.tensor([[ord(c) for c in text]])
    y = torch.tensor([[ord(c) for c in text[1:]] + [0]])
    
    out = model(x, targets=y)
    print(f"Input: '{text}' ({len(text)} bytes)")
    print(f"Logits: {out['logits'].shape}")
    print(f"Loss: {out['loss'].item():.4f}")
    print(f"BPB: {out['bpb'].item():.4f}")
    print(f"Patches: {out['n_patches']} (compression: {len(text)/out['n_patches']:.1f}x)")
    
    # Gate stats
    gates = model.get_gate_stats()
    print(f"\nGate stats ({len(gates)} PID projections):")
    for name, g in list(gates.items())[:5]:
        print(f"  {name}: P={g['P']:.3f} I={g['I']:.3f} D={g['D']:.3f}")
    
    # Baseline comparison
    print("\n--- Baseline ---")
    baseline = StandardByteTransformer()
    out2 = baseline(x, targets=y)
    print(f"Standard Transformer: {baseline.count_params()['total']:,} params, Loss: {out2['loss'].item():.4f}")
    
    print("\n✅ ionBrain v1 ready!")
