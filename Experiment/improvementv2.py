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

    Variant of `improvement.CustomAttention` with the attention-pattern term
    of the cross-block routing replaced by a *query-blocks* term. v1 built
    the coarse query/key from the block's causal running-average pattern, so
    the routing score was a blend of pattern-shape similarity and content
    similarity. Here that pattern term is gone: one small MLP per head
    compresses each block of queries into a single query-shaped vector, and
    the routing score is the similarity between an individual query position
    and those compressed block queries (Q vs. `q_block`). The content term
    (pooled `Kbar`) and its zero-initialized `content_scale` gate are kept
    exactly as in v1, so cross-block routing is
    `query-block similarity + gate * content similarity`, and both halves
    are combined with the same pre-scaled concatenation trick that lets
    SDPA run with a flat `scale=1.0`.
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

        G_max, L_max = block_geometry(config.block_size)
        span = max(G_max, L_max)
        head_dim = config.n_embd // config.n_head

        # Small per-head MLPs -- one for each head of this layer -- that
        # compress a whole block of queries (L_max * head_dim flattened)
        # down to a single query's dimension (head_dim), one vector per
        # block. These compressed block queries are the key side of the
        # *query-blocks* routing term: each individual query position of a
        # later block scores its similarity against them.
        compress_hidden = max(2 * head_dim, 16)
        self.compress = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(L_max * head_dim, compress_hidden, bias=config.bias),
                    nn.GELU(),
                    nn.Linear(compress_hidden, head_dim, bias=config.bias),
                )
                for _ in range(self.n_head)
            ]
        )
        self._L_max = L_max

        # Gate on the content-based routing term (the pooled `Kbar` half of
        # the cross-block query/key), exactly as in `improvement.py`.
        # Zero-initialized per head so training starts identical to
        # query-blocks-only routing and only leans on the content term once
        # it's shown to help -- the two terms live on different natural
        # scales (a pooled `Kbar` is an average over L keys, so its logits
        # run larger than the dot of a raw query with a compressed query).
        self.content_scale = nn.Parameter(torch.zeros(self.n_head))

        # Gate on the *entire* cross-block (flash-attention) branch, added on
        # top of the exact local output `P`. Zero-initialized per head for
        # the same warm-start reason as v1: at init this module is exactly
        # block-local causal attention, and cross-block information is phased
        # in only as training shows it helps.
        self.global_scale = nn.Parameter(torch.zeros(self.n_head))

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

        # per-block pooled pattern: the pooling weight that turns a block's
        # values into its single `Pbar` summary. Same derivation as v1: this
        # is just the last (whole-block) offset of the row-normalized causal
        # cumulative sum, which equals the full-block sum.
        summary_causal = local.cumsum(dim=-2)  # (B, H, G, L, L)
        summary_causal = summary_causal / summary_causal.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        summary = summary_causal[..., -1, :]  # (B, H, G, L)

        # content-based routing key for the cross-block step: pool K the
        # same way `summary` pools local-attention mass, so the coarse
        # routing score can compare real content, not just shape -- kept
        # from `improvement.py` unchanged.
        Kbar = torch.einsum('bhjk,bhjkd->bhjd', summary, K)  # (B, H, G, d)

        # pooled value per block (the thing cross-block queries actually
        # read). Its own dropout call, independent of the one applied to
        # `local` below, right alongside the other weights about to
        # multiply V.
        summary_v = self.attn_dropout(summary).to(V.dtype)
        Pbar = torch.einsum('bhjk,bhjkd->bhjd', summary_v, V)  # (B, H, G, d)

        # --- compressed block queries: the key side of the query-blocks
        # routing term. Flatten each block's queries (L * d per head) and
        # push it through that head's own per-layer MLP to get one
        # query-shaped vector per block. Padding to `_L_max` covers runs
        # where the runtime block length L is smaller than the
        # architecture's maximum.
        Q_flat = Q.reshape(Batch, self.n_head, G, L * head_dim)
        if L < self._L_max:
            Q_flat = F.pad(Q_flat, (0, self._L_max * head_dim - L * head_dim))
        q_flat_h = Q_flat.permute(1, 0, 2, 3)  # (H, B, G, L_max*d)
        q_block = torch.stack(
            [self.compress[h](q_flat_h[h]) for h in range(self.n_head)], dim=0
        ).permute(1, 0, 2, 3)  # (H, B, G, d) -> (B, H, G, d)

        # --- combined query-blocks + content query/key for the cross-block
        # step. Both halves now live in the query's dimension d (the pattern
        # half of v1 -- dim L -- is gone), so pre-scaling each half by
        # sqrt(sqrt(d)) makes a single dot product reproduce
        # Q.q_block/sqrt(d) + gate * Q.Kbar/sqrt(d) exactly, letting SDPA
        # run with a flat `scale=1.0` instead of its default 1/sqrt(E).
        gate = self.content_scale.view(1, self.n_head, 1, 1, 1)
        q_cross = torch.cat(
            [Q / (head_dim ** 0.25), (gate * Q) / (head_dim ** 0.25)], dim=-1
        )  # (B, H, G, L, 2*d)
        k_block = torch.cat(
            [q_block / (head_dim ** 0.25), Kbar / (head_dim ** 0.25)], dim=-1
        )  # (B, H, G, 2*d)

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

        q_cross = q_cross.reshape(Batch, self.n_head, G * L, 2 * head_dim).to(V.dtype)
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
