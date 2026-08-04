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
      P[b,h,i,l,:]  = local[b,h,i,l,:] @ V_local[b,h,i,:,:]  -- block i's own causal
                                                                 local output at offset l.
                                                                 V_local's key axis may be
                                                                 wider than block i alone (see
                                                                 the previous-block window in
                                                                 `forward`): local's last L
                                                                 columns are always block i's
                                                                 own causal weights, any
                                                                 columns before that are a
                                                                 fixed lookback into block i-1,
                                                                 which is unconditionally in
                                                                 the past so needs no extra
                                                                 causal masking beyond "block 0
                                                                 has no such lookback".
      Pbar[b,h,j,:] = summary[b,h,j,:] @ V_pool[b,h,j,:,:]   -- a single whole-block
                                                                 pooled value for block j

    glob is expected to already be masked to strictly j < i (row i=0 all zero).
    self_w is a *sigmoid* gate (not a softmax competitor): it is the
    probability mass block i's own causal output keeps for itself, and
    (1 - self_w) is the mass handed to the strictly-earlier blocks, split
    among them by their own softmax (glob.sum(-1) == 1 - self_w exactly,
    dropout aside). This is deliberately decoupled from how many earlier
    blocks exist -- see the "self-vs-earlier-blocks weighting" comment in
    `forward` for why. Block 0 has no earlier blocks to hand anything to,
    so self_w is forced to exactly 1 there, the correct degenerate case.

    Cross-block reads use Pbar (a full-block pooled summary) instead of
    P[j,l] (block j's causal output at the *querying* position's own offset
    l). Reading P[j,l] would mean a query at small l only ever sees the
    first l+1 positions of any earlier block j, no matter how far in the
    past j is -- an artificial bottleneck that has nothing to do with
    causality (block j is entirely in the past regardless of l). Pbar has
    no offset dependence, so every query that attends to block j sees the
    same complete pooled representation of it. `summary` (block j's own
    normalized local-attention mass over its positions -- see `forward`
    docstring) does double duty as both the pooling weights for Kbar (the
    content key used to score block j) and the pooling weights that build
    Pbar, mirroring how `local` doubles as both the own-block routing
    signal and the weights that build P.

    P and Pbar are recomputed in backward instead of saved from forward, so
    the module doesn't have to carry either the (B,H,G,L,d)-sized P or the
    (B,H,G,d)-sized Pbar across the forward/backward boundary.

    Shapes: local (B,H,G,L,K), glob (B,H,G,L,G), self_w (B,H,G,L),
            summary (B,H,G,L), V_local (B,H,G,K,d), V_pool (B,H,G,L,d),
            where K is L (window disabled) or 2L (window enabled).
    """

    @staticmethod
    def forward(ctx, local, glob, self_w, summary, V_local, V_pool):
        ctx.save_for_backward(local, glob, self_w, summary, V_local, V_pool)
        P = torch.matmul(local, V_local)                            # (B,H,G,L,d)
        Pbar = torch.einsum('bhjk,bhjkd->bhjd', summary, V_pool)    # (B,H,G,d)
        mixed = torch.einsum('bhilj,bhjd->bhild', glob, Pbar)
        own = self_w.unsqueeze(-1) * P
        return own + mixed

    @staticmethod
    def backward(ctx, grad_out):
        local, glob, self_w, summary, V_local, V_pool = ctx.saved_tensors
        d_local = d_glob = d_self_w = d_summary = d_V_local = d_V_pool = None

        need_P = ctx.needs_input_grad[0] or ctx.needs_input_grad[2] or ctx.needs_input_grad[4]
        need_Pbar = ctx.needs_input_grad[1]
        need_dPbar = ctx.needs_input_grad[3] or ctx.needs_input_grad[5]

        if need_P:
            P = torch.matmul(local, V_local)
        if need_Pbar:
            Pbar = torch.einsum('bhjk,bhjkd->bhjd', summary, V_pool)

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
            # dP carries only the own-block (self_w) term -- mixed no
            # longer reads P at all
            dP = self_w.unsqueeze(-1) * grad_out
            if ctx.needs_input_grad[0]:
                d_local = torch.matmul(dP, V_local.transpose(-2, -1))
            if ctx.needs_input_grad[4]:
                d_V_local = torch.matmul(local.transpose(-2, -1), dP)

        if ctx.needs_input_grad[3]:
            d_summary = torch.einsum('bhjd,bhjkd->bhjk', dPbar, V_pool)

        if ctx.needs_input_grad[5]:
            d_V_pool = torch.einsum('bhjk,bhjd->bhjkd', summary, dPbar)

        return d_local, d_glob, d_self_w, d_summary, d_V_local, d_V_pool


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

        # Gate on the content-based routing score computed in `forward`
        # (real Q.K similarity between a query and a pooled block
        # representation). Zero at init switches content-based routing off
        # entirely: global_scores (over strictly-earlier blocks) all start
        # at exactly 0, so once any probability mass reaches the earlier
        # blocks it starts out split *uniformly* among them; and the
        # content-based term inside self_scores also starts at 0, leaving
        # self_bias (below) as the sole driver of self-vs-earlier routing
        # at init. The gate opens up as training shows content-based
        # routing helps.
        self.content_scale = nn.Parameter(torch.zeros(self.n_head))

        # Per-head bias feeding the self-weight sigmoid (see "self-vs-
        # earlier-blocks weighting" in `forward`). Initialized large and
        # positive so self_weight ~= sigmoid(self_bias_init) regardless of
        # G, i.e. training starts near pure local (own-block) attention --
        # exactly as expressive as ordinary windowed attention -- and only
        # opens up cross-block mixing as content_scale and self_bias learn
        # it helps, instead of starting at self_weight = 1/(G+1) (diluted
        # by, and getting worse with, sequence length).
        self_bias_init = getattr(config, "self_bias_init", 4.0)
        self.self_bias = nn.Parameter(torch.full((self.n_head,), float(self_bias_init)))

        # Fixed causal lookback: block i's own local output also attends to
        # the entirety of block i-1 (unconditionally in the past), not just
        # block i. Without this, a query at small offset l has almost no
        # real local context (offset l=0 sees only itself), so the tokens
        # right before its block boundary -- usually the most relevant
        # recent context -- were only visible through a whole-block-
        # averaged Pbar. Cost is a second (B,H,G,L,L) matmul: same
        # complexity CLASS as before (O(T^1.5) total), just ~2x the
        # constant, since it widens the two blocks a query already reads
        # rather than adding more blocks to read.
        self.use_prev_block_window = getattr(config, "use_prev_block_window", True)

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
        scale = 1.0 / math.sqrt(head_dim)

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
        # This stays scoped to "this block only" on purpose: it's what
        # feeds summary/summary_causal/Kbar/Kbar_causal below, i.e. the
        # pooling & routing machinery that describes a block's own content
        # to the *rest* of the sequence. Widening it would blur that
        # description with borrowed context from the previous block.
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
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
        # is what keeps the pooled content key used by the self score (built
        # below) causally valid -- it may only depend on rows 0..l, never on
        # positions after l within the same block.
        summary_causal = local.cumsum(dim=-2)  # (B, H, G, L, L)
        # Normalize each L-sized (key-axis) vector back onto the simplex.
        # Raw cumsum grows with the query position l -- row l sums to l+1,
        # not 1 -- so without this, pooling with it would give positions
        # later in a block a systematically larger-magnitude pooled key,
        # purely from accumulated mass, not from anything content-related.
        # Dividing each vector by its own sum turns it into a running
        # *average* of the local attention pattern seen so far (0..l), so
        # scale no longer depends on l. clamp_min is just numerical-safety
        # padding: the true minimum sum is 1 (row l=0 always contributes
        # exactly one unit of mass), so it should never actually engage.
        summary_causal = summary_causal / summary_causal.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        # per-block summary, used as pooling weights to build the earlier
        # block's content key (Kbar, below) and pooled value (Pbar, in the
        # combine step); safe to pool over the whole block since such a
        # block is entirely in the past relative to any block that reads it.
        summary = local.sum(dim=-2)  # (B, H, G, L)
        # Same fix, same reason: the raw sum totals the block's real query
        # count (L for a full block, fewer for a padded final block), not 1.
        # Normalizing puts every block's pooling weights on the simplex
        # regardless of how many real positions fed it, so blocks aren't
        # implicitly weighted by their occupancy.
        summary = summary / summary.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        # --- own-block causal output, extended with a fixed lookback into
        # the immediately preceding block (see __init__ for the reasoning) ---
        if self.use_prev_block_window and G > 1:
            zero_kv = torch.zeros_like(K[:, :, :1])
            K_prev = torch.cat([zero_kv, K[:, :, :-1]], dim=2)          # (B,H,G,L,d)
            V_prev = torch.cat([torch.zeros_like(V[:, :, :1]), V[:, :, :-1]], dim=2)

            scores_prev = torch.matmul(Q, K_prev.transpose(-2, -1)) * scale  # (B,H,G,L,L)
            # Block i-1 is unconditionally in the past relative to block i,
            # so every position in it is a valid key for every offset in
            # block i -- no per-position triangular mask needed here, only
            # "does a previous block exist at all" (false only for i=0).
            has_prev = (torch.arange(G, device=X.device) > 0).view(G, 1, 1)
            scores_prev = scores_prev.masked_fill(~has_prev, float("-inf"))

            scores_ext = torch.cat([scores_prev, scores], dim=-1)  # (B,H,G,L,2L)
            local_ext = F.softmax(scores_ext, dim=-1)
            local_ext = torch.nan_to_num(local_ext, nan=0.0)  # defensive; every row keeps >=1 finite entry
            if padding:
                local_ext = local_ext * real.view(G, L, 1).to(local_ext.dtype)
            V_local = torch.cat([V_prev, V], dim=-2)  # (B,H,G,2L,d) -- concat on the per-block
                                                        # token axis to match scores_ext's key axis,
                                                        # NOT dim=2 (that's the block-shift axis
                                                        # used just above to build K_prev/V_prev)
        else:
            local_ext = local
            V_local = V

        # --- causal attention between block i's per-position query and
        # strictly earlier blocks, scored by real content similarity ---
        # (previously this compared the *shape* of local attention patterns
        # via summary_causal . summary -- i.e. whether two blocks
        # concentrated attention on the same relative offsets, which is a
        # positional coincidence, not a relevance signal, and lives on a
        # different scale than an ordinary dot product. Q.Kbar below is a
        # standard scaled-dot-product content score instead, pooling K the
        # same way the combine step pools V.)
        Kbar = torch.einsum('bhjk,bhjkd->bhjd', summary, K)  # (B,H,G,d) pooled content key, earlier blocks
        global_scores = torch.einsum('bhild,bhjd->bhilj', Q, Kbar) * scale

        gate = self.content_scale.view(1, self.n_head, 1, 1)  # (1,H,1,1), broadcasts over self_scores
        global_scores = global_scores * gate.unsqueeze(-1)  # (1,H,1,1,1), extra dim for the block axis

        # masked_fill overwrites (not multiplies), so gating before or
        # after masking is equally safe -- do it before, so masked entries
        # land on an exact -inf rather than a gated one.
        strict_allowed = self.causal_mask_strict[:G, :G]  # (G, G), j < i only
        global_scores = global_scores.masked_fill(
            ~strict_allowed.view(1, 1, G, 1, G), float("-inf")
        )

        # self score: real content similarity between a query and its own
        # block's causal pooled content key, same mechanism as the
        # cross-block score above but against Kbar_causal instead of Kbar.
        # We can't reuse the pooled `summary` to build this block's own
        # content key -- it sums over every query in the block, including
        # ones after position l, which would leak future tokens within the
        # block into position l's attention. `summary_causal` only
        # reflects rows 0..l, so pooling K with it stays causal.
        Kbar_causal = torch.matmul(summary_causal, K)  # (B,H,G,L,d)
        self_scores = torch.einsum('bhild,bhild->bhil', Q, Kbar_causal) * scale
        self_scores = self_scores * gate

        # --- self-vs-earlier-blocks weighting ---
        # Previously self and every earlier block competed in ONE softmax
        # over G+1 options, so at init (gate=0, all logits tied) self_weight
        # = 1/(G+1): ~97% of the output came from an unweighted average of
        # mostly-irrelevant pooled blocks at T=1024 (G~32), and this dilution
        # gets WORSE as T (and so G) grows. The model had to learn its way
        # out of a hole whose depth depends on sequence length, instead of
        # starting near pure local attention (exactly as expressive as
        # ordinary windowed attention) and learning *when* cross-block info
        # helps.
        #
        # Fix: decouple "how much goes to self" from "how the remainder is
        # split among earlier blocks". self_weight now comes from its own
        # sigmoid gate with a learned per-head bias (self_bias, init large
        # and positive), independent of G. The remaining probability mass
        # (1 - self_weight) is split among strictly-earlier blocks by their
        # own softmax. self_weight + glob.sum(-1) == 1 still holds exactly
        # (dropout aside) -- see below -- so this is still a valid convex
        # combination, it just starts at self_weight ~= sigmoid(self_bias)
        # regardless of how many blocks precede it.
        self_bias = self.self_bias.view(1, self.n_head, 1, 1)
        self_weight = torch.sigmoid(self_scores + self_bias)  # (B,H,G,L)

        glob_relative = F.softmax(global_scores, dim=-1)  # (B,H,G,L,G): split among earlier blocks
        glob_relative = torch.nan_to_num(glob_relative, nan=0.0)  # block 0: no earlier blocks -> all -inf row

        # Block 0 has no earlier blocks at all: force it to the exact
        # degenerate case (self_weight == 1) rather than whatever the
        # learned bias happens to produce, matching the original module's
        # invariant for this case.
        no_earlier = ~strict_allowed.any(dim=-1)  # (G,), True only for i == 0
        self_weight = self_weight.masked_fill(no_earlier.view(1, 1, G, 1), 1.0)

        glob = glob_relative * (1.0 - self_weight).unsqueeze(-1)  # (B,H,G,L,G)
        # self_weight + glob.sum(-1) == 1 everywhere: for i>0, glob_relative
        # sums to 1 so glob sums to (1 - self_weight); for i==0, glob_relative
        # is all zero (nan_to_num'd) and self_weight is forced to 1, so the
        # total is still exactly 1.

        # local_ext, glob, and self_weight are each per-query weights (not
        # shared across multiple future queries), so per-call dropout on
        # them is ordinary attention-weight regularization, same spirit as
        # the original module. `summary`, in contrast, is read by every
        # future query that attends to this block via Pbar -- it is no
        # longer given its own independent dropout call (previously
        # `summary_v = self.attn_dropout(summary)`), since that injected
        # non-regularizing, shared noise into a multi-consumer pooled
        # representation rather than per-query noise.
        local_ext = self.attn_dropout(local_ext)
        glob = self.attn_dropout(glob)
        self_weight = self.attn_dropout(self_weight)
        local_ext = local_ext.to(V.dtype)
        glob = glob.to(V.dtype)
        self_weight = self_weight.to(V.dtype)
        summary_pool = summary.to(V.dtype)

        y = _ScaledBlockCombine.apply(local_ext, glob, self_weight, summary_pool, V_local, V)

        y = y.permute(0, 2, 3, 1, 4).reshape(Batch, G * L, Embedding)
        if padding:
            y = y[:, :Token]
        y = self.c_proj(y)
        y = self.resid_dropout(y)
        return y