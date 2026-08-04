"""
Square-root blocked causal attention: local attention inside blocks, one global
attention *between* blocks, combined without ever building the full T x T map.

The sequence is cut into G blocks of L tokens each, with L = G = sqrt(T), so
for the usual T=1024 that is 32 blocks of 32 tokens. T must be a perfect
square.

  1. **Local.** Ordinary causal attention inside each block on its own:
     `local[g] = softmax(Q_g K_g^T / sqrt(head_dim))`, shape (G, L, L). Nothing
     crosses a block boundary here.
  2. **Summary.** Each local map is collapsed to one length-L vector by summing
     its rows together (`sum` over the *query* axis), i.e. "how much attention
     did each key position in this block receive". Stacking the G of them gives
     an (G, L) matrix - 32x32 in the running example.
     (Summing over the *key* axis instead would be useless: those rows are
     softmax outputs, so each one sums to exactly 1 and the summary would be a
     constant matrix of ones.)
  3. **Global.** That (G, L) matrix is treated as a sequence of G tokens with L
     features and self-attended causally: `glob = softmax(M M^T / sqrt(L))`,
     shape (G, G). No new projections - the summaries are their own Q and K.
  4. **Combine.** Own-block output is always included in full (weight 1, no
     softmax competition); *other* blocks are mixed in strictly by block index,
     scaled by that block's row of the global map:

         out[i] = local[i] @ V[i]  +  sum_{j<i} glob[i, j] * (local[j] @ V[j])

     folded so the (G, G, L, L) stack of scaled maps is never a real tensor.

V is untouched until the very last step, exactly as in the vanilla path.

Memory
------
Only `local` (B, H, G, L, L) and `glob` (B, H, G, G) are kept for backward, so
the attention activations are B*H*T*sqrt(T) instead of vanilla's B*H*T*T - a 32x
reduction at T=1024. `_ScaledBlockCombine` is a hand-written autograd Function
precisely so the scaled maps stay virtual: forward contracts straight through
them, and backward reconstructs what it needs from `local` and `glob` on the
fly. Nothing of size (G, G, L, L) is ever allocated in either direction.

Causality
---------
Token (i, l) reads block j<i in full and its own block up to position l, so the
value path is strictly causal.

An earlier version let block i attend to itself at the global level too
(`j <= i`), which leaked future tokens into the past: `glob[i, :]` is derived
from `summary[i]`, which sums over *every* query in block i (including queries
after position l), and that same glob row was then used to weight the block's
own local output for every position l in the block - so how much an early
token in the block "trusted" its own local attention depended on what later
tokens in that block did. The global softmax now only ever ranges over
strictly earlier blocks (`j < i`); the own-block contribution is added outside
that softmax with a fixed weight of 1, so it can no longer be modulated by a
future-influenced weight. Block 0 has no `j < 0`, so its global row is all
zero and its output is exactly `local[0] @ V[0]`.
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from .attention_registry import BaseAttention, register_attention


def block_geometry(T):
    """
    (number of blocks, block length) for a sequence of T tokens.

    T must be a perfect square: L = G = sqrt(T). T=1024 is not a perfect
    square and would raise; T=1089 (33^2) gives a 33x33 split.
    """
    L = math.isqrt(max(T, 0))
    if L * L != T:
        raise ValueError(f"sequence length {T} must be a perfect square")
    G = L
    return G, L


class _ScaledBlockCombine(torch.autograd.Function):
    """
    out[b,h,i,l,:] = P[b,h,i,l,:] + sum_{j<i} glob[b,h,i,j] * P[b,h,j,l,:]
    where P[b,h,j] = local[b,h,j] @ V[b,h,j].

    The own-block term is added with a fixed weight of 1, outside the glob
    softmax (glob's diagonal is always 0 - see the module docstring's
    Causality section for why). `glob` is expected to already be masked to
    strictly j < i (row 0 all zero).

    Written by hand rather than left to autograd so that the intermediate
    per-output-block scaled maps (`local[j] * glob[i][j]`, a (G, G, L, L) object)
    are never materialised, and so the per-block products `local @ V` are
    recomputed in backward instead of being stashed by the forward pass.

    Shapes: local (B, H, G, L, L), glob (B, H, G, G), V (B, H, G, L, d).
    """

    @staticmethod
    def forward(ctx, local, glob, V):
        ctx.save_for_backward(local, glob, V)
        # P[j] = local[j] @ V[j]: apply each block's local map to its own values
        # once, then let the global map mix the blocks. Contracting in this order
        # is what keeps the scaled maps virtual.
        P = torch.matmul(local, V)
        B, H, G, L, d = P.shape
        mixed = torch.matmul(glob, P.reshape(B, H, G, L * d)).view(B, H, G, L, d)
        return P + mixed

    @staticmethod
    def backward(ctx, grad_out):
        local, glob, V = ctx.saved_tensors
        B, H, G, L, d = grad_out.shape
        grad_flat = grad_out.reshape(B, H, G, L * d)

        d_local = d_glob = d_V = None

        if ctx.needs_input_grad[1]:
            # rebuild the per-block outputs from what was saved rather than
            # having carried a (B, H, G, L, d) tensor through the forward
            P = torch.matmul(local, V).reshape(B, H, G, L * d)
            d_glob = torch.matmul(grad_flat, P.transpose(-2, -1))

        if ctx.needs_input_grad[0] or ctx.needs_input_grad[2]:
            # gradient w.r.t. P: the direct residual term contributes grad_out
            # itself, and the mixed term contributes glob^T @ grad_out, same as
            # the plain glob@P case
            dP = grad_out + torch.matmul(glob.transpose(-2, -1), grad_flat).view(B, H, G, L, d)
            if ctx.needs_input_grad[0]:
                d_local = torch.matmul(dP, V.transpose(-2, -1))
            if ctx.needs_input_grad[2]:
                d_V = torch.matmul(local.transpose(-2, -1), dP)

        return d_local, d_glob, d_V


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

        # One triangular mask serves the local level: the local maps slice
        # [:L, :L] and need query l to see keys 0..l inclusive (ordinary
        # causal). Both G and L equal sqrt(block_size), so this is 32x32 for a
        # 1024-token context.
        G_max, L_max = block_geometry(config.block_size)
        span = max(G_max, L_max)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(span, span, dtype=torch.bool)),
            persistent=False,
        )
        # The global level uses a *strict* lower-triangular mask instead: block
        # i must not attend to itself here (see the Causality section above),
        # only to strictly earlier blocks j < i.
        self.register_buffer(
            "causal_mask_strict",
            torch.tril(torch.ones(span, span, dtype=torch.bool), diagonal=-1),
            persistent=False,
        )

    def forward(self, X):
        Batch, Token, Embedding = X.shape
        assert Token <= self.block_size, (
            f"sequence length {Token} exceeds block_size {self.block_size}"
        )
        head_dim = Embedding // self.n_head
        G, L = block_geometry(Token)

        Q = self.Wq(X)
        K = self.Wk(X)
        V = self.Wv(X)

        # (Batch, Token, Embedding) -> (Batch, n_head, G, L, head_dim)
        def blocks(t):
            return t.view(Batch, G, L, self.n_head, head_dim).permute(0, 3, 1, 2, 4)

        Q, K, V = blocks(Q), blocks(K), blocks(V)

        # --- 1. local attention, each block against itself only ---------------
        #! might need to fix the scaling factor here
        scores = torch.matmul(Q, K.transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))
        allowed = self.causal_mask[:L, :L]
        scores = scores.masked_fill(~allowed, float("-inf"))
        local = F.softmax(scores, dim=-1)

        # --- 2. one summary vector per block ---------------------------------
        # sum the rows of each local map together: entry m is the total
        # attention key m received from the queries in its block
        summary = local.sum(dim=-2)  # (Batch, n_head, G, L)

        # --- 3. attention between the blocks themselves -----------------------
        # strictly j < i: block i's own contribution is added back outside this
        # softmax (see forward's combine step and the Causality docstring section)
        #! might need to fix the scaling factor here
        global_scores = torch.matmul(summary, summary.transpose(-2, -1)) * (1.0 / math.sqrt(L))
        global_scores = global_scores.masked_fill(~self.causal_mask_strict[:G, :G], float("-inf"))
        glob = F.softmax(global_scores, dim=-1)  # (Batch, n_head, G, G)
        # block 0 has no j < 0: its row is all -inf, softmax gives NaN, zero it out
        glob = torch.nan_to_num(glob, nan=0.0)

        local = self.attn_dropout(local)
        glob = self.attn_dropout(glob)

        # Under autocast softmax returns fp32 while V is bf16; line the dtypes up
        # here so the hand-written backward (which runs outside autocast) sees a
        # single dtype, and so the big `local` tensor is saved at V's width.
        local = local.to(V.dtype)
        glob = glob.to(V.dtype)

        # --- 4. scale the local maps by the global row and apply to V ---------
        y = _ScaledBlockCombine.apply(local, glob, V)  # (Batch, n_head, G, L, head_dim)

        y = y.permute(0, 2, 3, 1, 4).reshape(Batch, G * L, Embedding)
        y = self.c_proj(y)
        y = self.resid_dropout(y)
        return y
    
    
    
    
# """
# Square-root blocked causal attention: local attention inside blocks plus one
# global attention between block summaries, computed without ever materializing
# the full T x T attention map.

# The global attention's query side is a *causal cumulative* summary (cumsum
# over the query axis, not a full-block sum), and blocks never attend to
# themselves at the global level -- both are required for strict causality;
# see the Causality note below `forward`.
# """

# import math

# import torch
# import torch.nn as nn
# from torch.nn import functional as F

# from .attention_registry import BaseAttention, register_attention


# def block_geometry(T):
#     """(G, L) with L = ceil(sqrt(T)), G = ceil(T / L)."""
#     L = math.isqrt(max(T - 1, 0)) + 1
#     G = (T + L - 1) // L
#     return G, L


# class _ScaledBlockCombine(torch.autograd.Function):
#     """
#     out[b,h,i,l,:] = P[b,h,i,l,:] + sum_{j<i} glob[b,h,i,l,j] * P[b,h,j,l,:]
#     where P[b,h,j] = local[b,h,j] @ V[b,h,j].

#     glob is expected to already be masked to strictly j < i (row i=0 all zero).
#     The own-block term (P[i,l,:]) is added directly, at full weight, outside
#     the glob softmax.

#     P is recomputed in backward instead of saved from forward, so the module
#     doesn't have to carry an extra (B,H,G,L,d)-sized tensor across the
#     forward/backward boundary.

#     Shapes: local (B,H,G,L,L), glob (B,H,G,L,G), V (B,H,G,L,d).
#     """

#     @staticmethod
#     def forward(ctx, local, glob, V):
#         ctx.save_for_backward(local, glob, V)
#         P = torch.matmul(local, V)  # P[j] = local[j] @ V[j]
#         mixed = torch.einsum('bhilj,bhjld->bhild', glob, P)
#         return P + mixed

#     @staticmethod
#     def backward(ctx, grad_out):
#         local, glob, V = ctx.saved_tensors
#         d_local = d_glob = d_V = None

#         need_P = ctx.needs_input_grad[0] or ctx.needs_input_grad[1] or ctx.needs_input_grad[2]
#         if need_P:
#             P = torch.matmul(local, V)

#         if ctx.needs_input_grad[1]:
#             d_glob = torch.einsum('bhild,bhjld->bhilj', grad_out, P)

#         if ctx.needs_input_grad[0] or ctx.needs_input_grad[2]:
#             # dP[j] gets the direct residual term (i==j) plus every mixed use of block j
#             dP = grad_out + torch.einsum('bhilj,bhild->bhjld', glob, grad_out)
#             if ctx.needs_input_grad[0]:
#                 d_local = torch.matmul(dP, V.transpose(-2, -1))
#             if ctx.needs_input_grad[2]:
#                 d_V = torch.matmul(local.transpose(-2, -1), dP)

#         return d_local, d_glob, d_V


# @register_attention("custom", "sqrt_block")
# class CustomAttention(BaseAttention):
#     """Drop-in replacement for `model.attention.Attention`; same in, same out."""

#     def __init__(self, config):
#         super().__init__(config)
#         assert config.n_embd % config.n_head == 0
#         self.Wq = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
#         self.Wk = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
#         self.Wv = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

#         self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
#         self.attn_dropout = nn.Dropout(config.dropout)
#         self.resid_dropout = nn.Dropout(config.dropout)
#         self.n_head = config.n_head
#         self.block_size = config.block_size

#         G_max, L_max = block_geometry(config.block_size)
#         span = max(G_max, L_max)
#         # local level: ordinary causal mask, j <= i (a query sees its own key)
#         self.register_buffer(
#             "causal_mask",
#             torch.tril(torch.ones(span, span, dtype=torch.bool)),
#             persistent=False,
#         )
#         # global level: STRICT causal mask, j < i (a block never attends to
#         # itself at the global level -- see Causality note below `forward`)
#         self.register_buffer(
#             "causal_mask_strict",
#             torch.tril(torch.ones(span, span, dtype=torch.bool), diagonal=-1),
#             persistent=False,
#         )

#     def forward(self, X):
#         Batch, Token, Embedding = X.shape
#         assert Token <= self.block_size
#         head_dim = Embedding // self.n_head
#         G, L = block_geometry(Token)
#         padding = G * L - Token

#         Q = self.Wq(X)
#         K = self.Wk(X)
#         V = self.Wv(X)

#         if padding:
#             Q = F.pad(Q, (0, 0, 0, padding))
#             K = F.pad(K, (0, 0, 0, padding))
#             V = F.pad(V, (0, 0, 0, padding))

#         def blocks(t):
#             return t.view(Batch, G, L, self.n_head, head_dim).permute(0, 3, 1, 2, 4)

#         Q, K, V = blocks(Q), blocks(K), blocks(V)

#         # --- local causal attention within each block ---
#         scores = torch.matmul(Q, K.transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))
#         allowed = self.causal_mask[:L, :L]
#         if padding:
#             real = (torch.arange(G * L, device=X.device) < Token).view(G, L)
#             allowed = allowed & real.unsqueeze(-2)
#         scores = scores.masked_fill(~allowed, float("-inf"))
#         local = F.softmax(scores, dim=-1)
#         if padding:
#             local = local * real.view(G, L, 1).to(local.dtype)

#         # per-position CAUSAL summary: cumulative sum over the query axis, so
#         # position l's summary only reflects rows 0..l of its own block. This
#         # is what makes the global query side causally valid (see note below).
#         summary_causal = local.cumsum(dim=-2)  # (B, H, G, L, L)
#         # per-block summary, used only as the KEY side for strictly later
#         # blocks; safe to pool over the whole block since such a block is
#         # entirely in the past relative to any block that reads it.
#         summary = local.sum(dim=-2)  # (B, H, G, L)

#         # --- causal attention between block i's per-position query and
#         # strictly earlier blocks' summaries; block i never attends to itself
#         # here (see Causality note below) ---
#         global_scores = torch.einsum('bhilm,bhjm->bhilj', summary_causal, summary)
#         global_scores = global_scores * (1.0 / math.sqrt(L))
#         strict_allowed = self.causal_mask_strict[:G, :G]  # (G, G), j < i only
#         global_scores = global_scores.masked_fill(
#             ~strict_allowed.view(1, 1, G, 1, G), float("-inf")
#         )
#         glob = F.softmax(global_scores, dim=-1)  # (B, H, G, L, G)
#         glob = torch.nan_to_num(glob, nan=0.0)  # block 0 has no j < 0

#         local = self.attn_dropout(local)
#         glob = self.attn_dropout(glob)
#         local = local.to(V.dtype)
#         glob = glob.to(V.dtype)

#         y = _ScaledBlockCombine.apply(local, glob, V)  # (B, H, G, L, head_dim)

#         y = y.permute(0, 2, 3, 1, 4).reshape(Batch, G * L, Embedding)
#         if padding:
#             y = y[:, :Token]
#         y = self.c_proj(y)
#         y = self.resid_dropout(y)
#         return y

# # Causality
# # ---------
# # Token (i, l) reads block j < i in full, and its own block up to position l,
# # so the value path is strictly causal. The global *weights* are causal too,
# # by construction of the two changes above:
# #   - blocks never attend to themselves at the global level (`causal_mask_strict`,
# #     j < i), so a block's own future-pooled content can never modulate its
# #     own output; and
# #   - the query side (`summary_causal`) is a running cumulative sum, so
# #     position l's mixing weights over past blocks are built only from rows
# #     0..l of its own block's local attention map, never from rows > l.
# # Together these remove both causality leaks discussed for earlier versions
# # of this module: (1) a block's own future tokens modulating trust in its own
# # local output via a self-referencing global weight, and (2) a block's future
# # tokens modulating the mixing weight used for a strictly-past block, via a
# # full-block (rather than causal, cumulative) query-side summary.