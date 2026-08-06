import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from .attention_registry import BaseAttention, register_attention


def block_geometry(T):
    """(G, L) with L = ceil(sqrt(T)), G = ceil(T / L)."""
    L = math.isqrt(max(T - 1, 0)) + 1
    G = (T + L - 1) // L
    return G, L


@register_attention("custom", "sqrt_block")
class CustomAttention(BaseAttention):
    """Drop-in replacement for `model.attention.Attention`; same in, same out.

    Two-level attention: an exact causal `local` pass inside each block, and
    a coarse `cross`-block pass that lets each position attend to strictly
    earlier blocks' pooled (summary, content) representations. The cross
    pass is delegated to `F.scaled_dot_product_attention` (flash attention),
    and its output is mixed into the exact local output through a
    zero-initialized per-head gate (`global_scale`), so training starts
    identical to plain block-local causal attention and only leans on
    cross-block information once it's shown to help.

    Earlier versions of this module ran the cross-block step by hand
    (materializing a (B,H,G,L,G) score tensor, softmaxing it jointly with a
    hand-rolled "self" score so the own block competed for a share of the
    same normalization) and used a custom autograd.Function to avoid
    carrying two extra tensors across the forward/backward boundary. Once
    that step is handed to a fused attention kernel, its softmax is opaque
    -- there's no logit or log-sum-exp to jointly normalize against an
    external "self" term -- so the own block's contribution is instead just
    `P` (the exact local output) added directly, and the custom
    autograd.Function is gone too: SDPA already has its own efficient
    backward, and `P`/`cross` are now independent tensors combined with a
    plain add, which autograd handles on its own.
    """

    def __init__(self, config):
        super().__init__(config)
        assert config.n_embd % config.n_head == 0
        self.Wq = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.Wk = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.Wv = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.block_size = config.block_size

        # Gate on the content-based routing term folded into the cross-block
        # query/key (real Q.K similarity between a query and a pooled block
        # representation, on top of the pattern-shape score). Zero-initialized
        # per head so training starts identical to pattern-only routing and
        # only leans on content once it's shown to help -- the two terms live
        # on very different natural scales (content logits ~O(1) like
        # ordinary attention scores; pattern logits are dot products of
        # L-simplex vectors, typically << 1), so an ungated sum would let
        # content drown out the pattern signal from the very first step.
        self.content_scale = nn.Parameter(torch.zeros(self.n_head))

        # Gate on the *entire* cross-block (flash-attention) branch, added on
        # top of the exact local output `P`. Zero-initialized per head for
        # the same warm-start reason as `content_scale`: at init this module
        # is exactly block-local causal attention, and cross-block
        # information is phased in only as training shows it helps.
        self.global_scale = nn.Parameter(torch.zeros(self.n_head))

        G_max, L_max = block_geometry(config.block_size)
        span = max(G_max, L_max)
        # local level: ordinary causal mask, j <= i (a query sees its own key)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(span, span, dtype=torch.bool)),
            persistent=False,
        )
        # cross-block level: STRICT causal mask, j < i (a block never attends
        # to itself via the pooled summary -- its own contribution comes
        # entirely from the exact local pass instead; see class docstring)
        self.register_buffer(
            "causal_mask_strict",
            torch.tril(torch.ones(span, span, dtype=torch.bool), diagonal=-1),
            persistent=False,
        )

    def forward(self, X):
        Batch, Token, Embedding = X.shape
        assert Token <= self.block_size
        head_dim = Embedding // self.n_head
        G, L = block_geometry(Token)
        padding = G * L - Token

        Q = self.Wq(X)
        K = self.Wk(X)
        V = self.Wv(X)

        if padding:
            Q = F.pad(Q, (0, 0, 0, padding))
            K = F.pad(K, (0, 0, 0, padding))
            V = F.pad(V, (0, 0, 0, padding))

        def blocks(t):
            return t.view(Batch, G, L, self.n_head, head_dim).permute(0, 3, 1, 2, 4)

        Q, K, V = blocks(Q), blocks(K), blocks(V)

        # --- local causal attention within each block: the own-block
        # output, exact and fully causal, no pooling involved ---
        scores = torch.matmul(Q, K.transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))
        allowed = self.causal_mask[:L, :L]
        if padding:
            real = (torch.arange(G * L, device=X.device) < Token).view(G, L)
            allowed = allowed & real.unsqueeze(-2)
        scores = scores.masked_fill(~allowed, float("-inf"))
        local = F.softmax(scores, dim=-1)
        if padding:
            local = local * real.view(G, L, 1).to(local.dtype)

        # per-position causal running-average pattern: cumulative sum over
        # the query axis, so position l's summary only reflects rows 0..l of
        # its own block. This is the query side of the cross-block step
        # below, and it's what keeps that step causally valid.
        summary_causal = local.cumsum(dim=-2)  # (B, H, G, L, L)
        summary_causal = summary_causal / summary_causal.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        # per-block pooled pattern: the key side of the cross-block step,
        # one vector per block regardless of query offset (safe, since a
        # block being read this way is entirely in the past). This is just
        # the last (whole-block) offset of summary_causal, not a fresh
        # reduction over `local` -- see the note in `_ScaledBlockCombine`'s
        # historical version of this file for why that offset already
        # equals the full-block sum.
        summary = summary_causal[..., -1, :]  # (B, H, G, L)

        # content-based routing key for the cross-block step: pool K the
        # same way `summary` pools local-attention mass, so the coarse
        # routing score can compare real content, not just attention shape.
        Kbar = torch.einsum('bhjk,bhjkd->bhjd', summary, K)  # (B, H, G, d)

        # --- combined pattern+content query/key for the cross-block step ---
        # Pre-scaling each half by its own dimension so a single dot product
        # reproduces pattern_score/sqrt(L) + gate*content_score/sqrt(d)
        # exactly lets us hand SDPA a flat `scale=1.0` instead of its default
        # 1/sqrt(E) (which would apply one blended scale to both halves).
        gate = self.content_scale.view(1, self.n_head, 1, 1, 1)
        q_cross = torch.cat(
            [summary_causal / (L ** 0.25), (gate * Q) / (head_dim ** 0.25)], dim=-1
        )  # (B, H, G, L, L + d)
        k_block = torch.cat(
            [summary / (L ** 0.25), Kbar / (head_dim ** 0.25)], dim=-1
        )  # (B, H, G, L + d)

        # pooled value per block (the thing cross-block queries actually
        # read): summary is the pooling weight, same role `local` plays for
        # the own-block output above. Its own dropout call, independent of
        # the one applied to `local` below, right alongside the other
        # weights about to multiply V.
        summary_v = self.attn_dropout(summary).to(V.dtype)
        Pbar = torch.einsum('bhjk,bhjkd->bhjd', summary_v, V)  # (B, H, G, d)

        # Strictly-causal block mask (j < i), plus one fallback "null" key
        # reachable only from block 0's queries. Block 0 has no earlier
        # blocks, so its mask row would otherwise be all-False -> NaN
        # softmax; the null key carries a zero value, so block 0's
        # cross-block contribution comes out exactly zero, matching the
        # degenerate case handled by the old joint softmax.
        strict_allowed = self.causal_mask_strict[:G, :G]  # (G, G)
        block0_only = torch.zeros(G, 1, dtype=torch.bool, device=X.device)
        block0_only[0, 0] = True
        key_mask = torch.cat([strict_allowed, block0_only], dim=-1)  # (G, G+1)
        key_mask = key_mask.unsqueeze(1).expand(G, L, G + 1).reshape(G * L, G + 1)

        q_cross = q_cross.reshape(Batch, self.n_head, G * L, L + head_dim).to(V.dtype)
        k_block_ext = torch.cat([k_block, torch.zeros_like(k_block[:, :, :1, :])], dim=2).to(V.dtype)
        Pbar_ext = torch.cat([Pbar, torch.zeros_like(Pbar[:, :, :1, :])], dim=2)

        cross = F.scaled_dot_product_attention(
            q_cross,
            k_block_ext,
            Pbar_ext,
            attn_mask=key_mask,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            scale=1.0,
        )  # (B, H, G*L, d)
        cross = cross.view(Batch, self.n_head, G, L, head_dim)

        local = self.attn_dropout(local)
        local = local.to(V.dtype)
        P = torch.matmul(local, V)  # (B, H, G, L, d) exact, own-block causal output

        out_gate = self.global_scale.view(1, self.n_head, 1, 1, 1)
        y = P + out_gate * cross

        y = y.permute(0, 2, 3, 1, 4).reshape(Batch, G * L, Embedding)
        if padding:
            y = y[:, :Token]
        y = self.c_proj(y)
        y = self.resid_dropout(y)
        return y