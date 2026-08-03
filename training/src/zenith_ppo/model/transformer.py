"""GQA causal decoder with RMSNorm, RoPE, SwiGLU, and SDPA/eager parity."""

from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F


def _rope_values(tokens, head_dim, device, dtype):
    positions = torch.arange(tokens, device=device, dtype=torch.float32)
    frequencies = torch.exp(torch.arange(0, head_dim, 2, device=device, dtype=torch.float32)
                            * (-math.log(10_000.0) / head_dim))
    angle = positions[:, None] * frequencies[None]
    return angle.cos().to(dtype), angle.sin().to(dtype)


def _rope(x, values=None):
    head_dim = x.shape[-1]
    if head_dim % 2:
        raise ValueError("RoPE head dimension must be even")
    cos, sin = values or _rope_values(x.shape[-2], head_dim, x.device, x.dtype)
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)


def _attention_layout(lengths, tokens, device):
    causal = torch.ones(tokens, tokens, dtype=torch.bool, device=device).tril()
    if lengths is None:
        return causal, None
    valid = torch.arange(tokens, device=device)[None] < lengths[:, None]
    valid_query = valid[:, None, :, None]
    mask = causal[None, None] & valid[:, None, None, :]
    # Padded queries are discarded after each block, but must still have one finite
    # softmax entry so NaNs cannot poison gradients through masked matrix products.
    first_key = torch.zeros_like(mask)
    first_key[..., 0] = True
    return (mask & valid_query) | (first_key & ~valid_query), valid


class CausalGQA(nn.Module):
    def __init__(self, d_model, query_heads, kv_heads, head_dim):
        super().__init__()
        if query_heads % kv_heads:
            raise ValueError("kv_heads must divide query_heads")
        self.qh, self.kvh, self.d = query_heads, kv_heads, head_dim
        self.qkv = nn.Linear(d_model, (query_heads + 2 * kv_heads) * head_dim, bias=False)
        self.out = nn.Linear(query_heads * head_dim, d_model, bias=False)

    def forward(self, x, lengths=None, backend="sdpa", *, rope=None, attention_mask=None):
        if backend not in {"eager", "sdpa"}:
            raise ValueError(f"unsupported attention backend {backend!r}")
        batch, tokens, _ = x.shape
        def shape(value, heads):
            return value.view(batch, tokens, heads, self.d).transpose(1, 2)
        q_raw, k_raw, v_raw = self.qkv(x).split(
            (self.qh * self.d, self.kvh * self.d, self.kvh * self.d), dim=-1
        )
        rope = rope or _rope_values(tokens, self.d, x.device, x.dtype)
        q, k, v = _rope(shape(q_raw, self.qh), rope), _rope(shape(k_raw, self.kvh), rope), shape(v_raw, self.kvh)
        repeat = self.qh // self.kvh
        mask = attention_mask
        if backend == "eager":
            # The reference path materializes repeated heads for transparent
            # parity testing.  SDPA below lets the fused kernel broadcast GQA
            # heads directly and avoids the K/V allocation.
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)
            if mask is None:
                mask, _ = _attention_layout(lengths, tokens, x.device)
            scores = (q.float() @ k.float().transpose(-2, -1)) / math.sqrt(self.d)
            scores = scores.masked_fill(~mask, -torch.inf)
            weights = torch.softmax(scores, -1).to(v.dtype)
            value = weights @ v
        else:
            if mask is None and lengths is not None:
                mask, _ = _attention_layout(lengths, tokens, x.device)
            value = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                dropout_p=0.0, is_causal=mask is None, enable_gqa=repeat > 1)
        return self.out(value.transpose(1, 2).reshape(batch, tokens, self.qh * self.d))


class DecoderBlock(nn.Module):
    def __init__(self, d_model, query_heads, kv_heads, head_dim, ffn_dim):
        super().__init__()
        self.n1, self.n2 = nn.RMSNorm(d_model), nn.RMSNorm(d_model)
        self.attention = CausalGQA(d_model, query_heads, kv_heads, head_dim)
        self.gate = nn.Linear(d_model, 2 * ffn_dim, bias=False)
        self.down = nn.Linear(ffn_dim, d_model, bias=False)

    def forward(self, x, lengths=None, backend="sdpa", *, rope=None,
                attention_mask=None, valid=None):
        x = x + self.attention(self.n1(x), lengths, backend, rope=rope,
                               attention_mask=attention_mask)
        gate, value = self.gate(self.n2(x)).chunk(2, dim=-1)
        x = x + self.down(F.silu(gate) * value)
        if valid is not None:
            x = torch.where(valid[..., None], x, torch.zeros((), dtype=x.dtype, device=x.device))
        return x


class Decoder(nn.Module):
    def __init__(self, *, layers=6, d_model=256, query_heads=8, kv_heads=2,
                 head_dim=32, ffn_dim=768, context_tokens=4096):
        super().__init__()
        if d_model != query_heads * head_dim:
            raise ValueError("d_model must equal query_heads * head_dim")
        self.blocks = nn.ModuleList(DecoderBlock(d_model, query_heads, kv_heads, head_dim, ffn_dim)
                                    for _ in range(layers))
        self.norm = nn.RMSNorm(d_model)
        cos, sin = _rope_values(context_tokens, head_dim, torch.device("cpu"), torch.float32)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x, lengths=None, backend="sdpa"):
        tokens = x.shape[1]
        rope = (self.rope_cos[:tokens].to(x.dtype), self.rope_sin[:tokens].to(x.dtype))
        if lengths is None:
            attention_mask, valid = None, None
        else:
            attention_mask, valid = _attention_layout(lengths, tokens, x.device)
        for block in self.blocks:
            x = block(x, lengths, backend, rope=rope, attention_mask=attention_mask, valid=valid)
        return self.norm(x)
