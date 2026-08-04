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


class _ScaledBlockCombine(torch.autograd.Function):
    """
    out[b,h,i,l,:] = self_w[b,h,i,l] * P[b,h,i,l,:] + sum_{j<i} glob[b,h,i,l,j] * Pbar[b,h,j,:]

    where:
      P[b,h,j,l,:]  = local[b,h,j,l,:]  @ V[b,h,j,:,:]   -- block j's own causal
                                                             local output at offset l
      Pbar[b,h,j,:] = summary[b,h,j,:] @ V[b,h,j,:,:]    -- a single whole-block
                                                             pooled value for block j

    glob is expected to already be masked to strictly j < i (row i=0 all zero).
    self_w is the softmax weight the own block earns by competing against the
    strictly-earlier blocks in the *same* normalization: for i>0,
    glob[i,l,:].sum() + self_w[i,l] == 1 (up to dropout), so the own block no
    longer gets an unconditional, unnormalized weight of 1 on top of the
    other blocks' share. Block 0 has no earlier blocks to compete with, so
    self_w == 1 there, which is the correct degenerate case.

    Cross-block reads use Pbar (a full-block pooled summary) instead of
    P[j,l] (block j's causal output at the *querying* position's own offset
    l). Reading P[j,l] would mean a query at small l only ever sees the
    first l+1 positions of any earlier block j, no matter how far in the
    past j is -- an artificial bottleneck that has nothing to do with
    causality (block j is entirely in the past regardless of l). Pbar has
    no offset dependence, so every query that attends to block j sees the
    same complete pooled representation of it. `summary` (block j's own
    normalized local-attention mass over its positions -- see `forward`
    docstring) does double duty as both the routing key for j and the
    pooling weights that build Pbar, mirroring how `local` doubles as both
    the own-block routing signal and the weights that build P.

    P and Pbar are recomputed in backward instead of saved from forward, so
    the module doesn't have to carry either the (B,H,G,L,d)-sized P or the
    (B,H,G,d)-sized Pbar across the forward/backward boundary.

    Shapes: local (B,H,G,L,L), glob (B,H,G,L,G), self_w (B,H,G,L),
            summary (B,H,G,L), V (B,H,G,L,d).
    """

    @staticmethod
    def forward(ctx, local, glob, self_w, summary, V):
        ctx.save_for_backward(local, glob, self_w, summary, V)
        P = torch.matmul(local, V)                                 # (B,H,G,L,d)
        Pbar = torch.einsum('bhjk,bhjkd->bhjd', summary, V)         # (B,H,G,d)
        mixed = torch.einsum('bhilj,bhjd->bhild', glob, Pbar)
        own = self_w.unsqueeze(-1) * P
        return own + mixed

    @staticmethod
    def backward(ctx, grad_out):
        local, glob, self_w, summary, V = ctx.saved_tensors
        d_local = d_glob = d_self_w = d_summary = d_V = None

        need_P = ctx.needs_input_grad[0] or ctx.needs_input_grad[2] or ctx.needs_input_grad[4]
        need_Pbar = ctx.needs_input_grad[1]
        need_dPbar = ctx.needs_input_grad[3] or ctx.needs_input_grad[4]

        if need_P:
            P = torch.matmul(local, V)
        if need_Pbar:
            Pbar = torch.einsum('bhjk,bhjkd->bhjd', summary, V)

        if ctx.needs_input_grad[2]:
            d_self_w = torch.einsum('bhild,bhild->bhil', grad_out, P)

        if ctx.needs_input_grad[1]:
            d_glob = torch.einsum('bhild,bhjd->bhilj', grad_out, Pbar)

        dPbar = None
        if need_dPbar:
            # gradient into block j's pooled value, summed over every (i,l)
            # that attended to it
            dPbar = torch.einsum('bhilj,bhild->bhjd', glob, grad_out)  # (B,H,G,d)

        if ctx.needs_input_grad[0] or ctx.needs_input_grad[4]:
            # dP now carries only the own-block (self_w) term -- mixed no
            # longer reads P at all, so there's no glob-weighted term here
            # the way there used to be
            dP = self_w.unsqueeze(-1) * grad_out
            if ctx.needs_input_grad[0]:
                d_local = torch.matmul(dP, V.transpose(-2, -1))
            if ctx.needs_input_grad[4]:
                d_V = torch.matmul(local.transpose(-2, -1), dP)
                d_V = d_V + torch.einsum('bhjk,bhjd->bhjkd', summary, dPbar)

        if ctx.needs_input_grad[3]:
            d_summary = torch.einsum('bhjd,bhjkd->bhjk', dPbar, V)

        return d_local, d_glob, d_self_w, d_summary, d_V


@register_attention("custom", "sqrt_block")
class CustomAttention(BaseAttention):
    """Drop-in replacement for `model.attention.Attention`; same in, same out."""

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

        # Gate on the content-based routing term added in `forward` (real
        # Q.K similarity between a query and a pooled block representation,
        # on top of the pattern-shape score). Zero-initialized per head so
        # training starts identical to pattern-only routing and only leans
        # on content once it's shown to help -- the two terms live on very
        # different natural scales (content logits ~O(1) like ordinary
        # attention scores; pattern logits are dot products of L-simplex
        # vectors, typically << 1), so an ungated sum would let content
        # drown out the pattern signal from the very first step.
        self.content_scale = nn.Parameter(torch.zeros(self.n_head))

        G_max, L_max = block_geometry(config.block_size)
        span = max(G_max, L_max)
        # local level: ordinary causal mask, j <= i (a query sees its own key)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(span, span, dtype=torch.bool)),
            persistent=False,
        )
        # global level: STRICT causal mask, j < i (a block never attends to
        # itself via the *pooled* summary -- see Causality note below
        # `forward`; the own block instead competes via a separate,
        # causally-valid self score)
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

        # --- local causal attention within each block ---
        scores = torch.matmul(Q, K.transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))
        allowed = self.causal_mask[:L, :L]
        if padding:
            real = (torch.arange(G * L, device=X.device) < Token).view(G, L)
            allowed = allowed & real.unsqueeze(-2)
        scores = scores.masked_fill(~allowed, float("-inf"))
        local = F.softmax(scores, dim=-1)
        if padding:
            local = local * real.view(G, L, 1).to(local.dtype)

        # per-position CAUSAL summary: cumulative sum over the query axis, so
        # position l's summary only reflects rows 0..l of its own block. This
        # is what makes the global query side (and the self score below)
        # causally valid.
        summary_causal = local.cumsum(dim=-2)  # (B, H, G, L, L)
        # Normalize each L-sized (key-axis) vector back onto the simplex.
        # Raw cumsum grows with the query position l -- row l sums to l+1,
        # not 1 -- so without this, positions later in a block would pick up
        # systematically larger dot products below purely from accumulated
        # magnitude, not from any real similarity. Dividing each vector by
        # its own sum turns it into a running *average* of the local
        # attention pattern seen so far (0..l), so scale no longer depends
        # on l. clamp_min is just numerical-safety padding: the true minimum
        # sum is 1 (row l=0 always contributes exactly one unit of mass), so
        # it should never actually engage.
        summary_causal = summary_causal / summary_causal.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        # per-block summary, used only as the KEY side for strictly earlier
        # blocks; safe to pool over the whole block since such a block is
        # entirely in the past relative to any block that reads it.
        summary = local.sum(dim=-2)  # (B, H, G, L)
        # Same fix, same reason: the raw sum totals the block's real query
        # count (L for a full block, fewer for a padded final block), not 1.
        # Normalizing puts every block's summary key on the simplex
        # regardless of how many real positions fed it, so blocks aren't
        # implicitly weighted by their occupancy.
        summary = summary / summary.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        # --- causal attention between block i's per-position query and
        # strictly earlier blocks' summaries ---
        global_scores = torch.einsum('bhilm,bhjm->bhilj', summary_causal, summary)
        global_scores = global_scores * (1.0 / math.sqrt(L))
        strict_allowed = self.causal_mask_strict[:G, :G]  # (G, G), j < i only
        global_scores = global_scores.masked_fill(
            ~strict_allowed.view(1, 1, G, 1, G), float("-inf")
        )

        # self score: lets block i's own (already-causal) local output
        # compete for its share of the mix instead of being added back
        # unconditionally at weight 1. We can't reuse the pooled `summary`
        # as the key here -- it sums over every query in the block,
        # including ones after position l, which would leak future tokens
        # within the block into position l's attention. `summary_causal`
        # dotted with itself only reflects rows 0..l, so it stays causal.
        self_scores = torch.einsum('bhilm,bhilm->bhil', summary_causal, summary_causal)
        self_scores = self_scores * (1.0 / math.sqrt(L))

        # --- content-based routing term ---
        # global_scores / self_scores above only compare *shapes* of local
        # attention patterns (summary, summary_causal); two blocks with
        # unrelated content but similarly-shaped local attention look
        # identical to that score, and related blocks with differently
        # shaped attention look unrelated. Add real Q.K content similarity
        # on top, pooling K the same way fix #1 pools V -- reusing
        # `summary`/`summary_causal` as pooling weights so no new routing
        # tensors are needed, just K instead of V.
        Kbar = torch.einsum('bhjk,bhjkd->bhjd', summary, K)  # (B,H,G,d) content key, earlier blocks
        content_global = torch.einsum('bhild,bhjd->bhilj', Q, Kbar) * (1.0 / math.sqrt(head_dim))

        Kbar_causal = torch.matmul(summary_causal, K)  # (B,H,G,L,d) causal content key, own block
        content_self = torch.einsum('bhild,bhild->bhil', Q, Kbar_causal) * (1.0 / math.sqrt(head_dim))

        gate = self.content_scale.view(1, self.n_head, 1, 1)  # (1,H,1,1), broadcasts over self_scores
        content_self = content_self * gate
        content_global = content_global * gate.unsqueeze(-1)  # (1,H,1,1,1), extra dim for the block axis

        # add after masking global_scores, so strictly-later blocks (still
        # -inf) stay -inf regardless of what content_global says about them
        global_scores = global_scores + content_global
        self_scores = self_scores + content_self

        # own block joins the same softmax as strictly-earlier blocks
        all_scores = torch.cat([global_scores, self_scores.unsqueeze(-1)], dim=-1)  # (B,H,G,L,G+1)
        glob_all = F.softmax(all_scores, dim=-1)
        glob_all = torch.nan_to_num(glob_all, nan=0.0)  # safety net; block 0 no longer needs it

        glob = glob_all[..., :G]        # (B, H, G, L, G) weight on strictly earlier blocks
        self_weight = glob_all[..., G]  # (B, H, G, L)    weight on own block

        # `summary` now feeds Pbar = summary @ V inside the combine function
        # (see below), i.e. it directly pools V the same way `local` does
        # for the own-block term -- so it gets its own independent dropout
        # mask here too, right alongside the other weights that are about
        # to multiply V. This is a fresh call to the same dropout module,
        # not a reuse of anything computed above.
        local = self.attn_dropout(local)
        glob = self.attn_dropout(glob)
        self_weight = self.attn_dropout(self_weight)
        summary_v = self.attn_dropout(summary)
        local = local.to(V.dtype)
        glob = glob.to(V.dtype)
        self_weight = self_weight.to(V.dtype)
        summary_v = summary_v.to(V.dtype)

        # cross-block reads now go through Pbar = summary_v @ V (a whole-block
        # pooled value), not P[j, l] (block j's own output at the querying
        # position's offset l) -- see _ScaledBlockCombine docstring.
        y = _ScaledBlockCombine.apply(local, glob, self_weight, summary_v, V)  # (B, H, G, L, head_dim)

        y = y.permute(0, 2, 3, 1, 4).reshape(Batch, G * L, Embedding)
        if padding:
            y = y[:, :Token]
        y = self.c_proj(y)
        y = self.resid_dropout(y)
        return y