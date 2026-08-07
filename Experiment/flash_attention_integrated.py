import math
import random

import torch
import torch.nn as nn
from torch.nn import functional as F

import triton
import triton.language as tl

from .attention_registry import BaseAttention, register_attention


def block_geometry(T):
    """(G, L) with L = ceil(sqrt(T)), G = ceil(T / L)."""
    L = math.isqrt(max(T - 1, 0)) + 1
    G = (T + L - 1) // L
    return G, L


# =====================================================================
# Streaming local-causal attention (Triton)
#
# Replaces the old "materialize (L, L) probs, use them twice" local pass
# with a kernel that walks one query row at a time. For each row i it:
#   1. computes that row's causal softmax,
#   2. immediately contracts it against V into P[i] (dropout applied),
#   3. immediately folds it into a running length-L accumulator `S`
#      that becomes summary_causal[i] once divided by the right count,
#   4. throws the row away.
#
# The only state carried between rows is `S` (shape (L,)) plus the K/V
# block loaded once at the top -- the (L, L) matrix of probabilities
# never exists as a tensor, on-chip or in HBM.
#
# Because every query row's softmax is over a single K block that's
# already resident (L is small -- it's ~sqrt(T)), there's no need for
# a second "tile over keys" loop the way ordinary FlashAttention needs
# for long sequences; the only sequential axis is the query row, which
# is unrolled inside one Triton program per (batch, head, block).
#
# Backward recomputes each row's softmax from Q/K (cheap, since nothing
# was saved) walking rows in *reverse*, maintaining a running suffix-sum
# `R` of d(summary_causal)/dS and running (L, d) accumulators for dK/dV
# -- the mirror image of the forward pass's running `S`.
# =====================================================================


@triton.jit
def _local_fwd_kernel(
    Q, K, V, P, Summary,
    stride_qb, stride_qh, stride_qg, stride_ql, stride_qd,
    stride_kb, stride_kh, stride_kg, stride_kl, stride_kd,
    stride_vb, stride_vh, stride_vg, stride_vl, stride_vd,
    stride_pb, stride_ph, stride_pg, stride_pl, stride_pd,
    stride_sb, stride_sh, stride_sg, stride_si, stride_sj,
    Token, L, G, H,
    softmax_scale, dropout_p, seed,
    HEAD_DIM: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_L: tl.constexpr,
):
    pid = tl.program_id(0)
    g = pid % G
    t = pid // G
    h = t % H
    b = t // H

    n_valid = tl.maximum(tl.minimum(L, Token - g * L), 0)

    offs_l = tl.arange(0, BLOCK_L)
    offs_d = tl.arange(0, BLOCK_D)
    mask_l = offs_l < L
    mask_d = offs_d < HEAD_DIM
    kv_mask = mask_l[:, None] & mask_d[None, :]

    qb = Q + b * stride_qb + h * stride_qh + g * stride_qg
    kb = K + b * stride_kb + h * stride_kh + g * stride_kg
    vb = V + b * stride_vb + h * stride_vh + g * stride_vg
    pb = P + b * stride_pb + h * stride_ph + g * stride_pg
    sb = Summary + b * stride_sb + h * stride_sh + g * stride_sg

    Kb = tl.load(kb + offs_l[:, None] * stride_kl + offs_d[None, :] * stride_kd,
                 mask=kv_mask, other=0.0).to(tl.float32)
    Vb = tl.load(vb + offs_l[:, None] * stride_vl + offs_d[None, :] * stride_vd,
                 mask=kv_mask, other=0.0).to(tl.float32)

    S = tl.zeros((BLOCK_L,), dtype=tl.float32)

    for i in range(BLOCK_L):
        i_in_range = i < L
        i_valid = i < n_valid

        q_i = tl.load(qb + i * stride_ql + offs_d * stride_qd,
                      mask=mask_d & i_in_range, other=0.0).to(tl.float32)

        row_mask = (offs_l <= i) & mask_l & i_valid
        scores = tl.sum(q_i[None, :] * Kb, axis=1) * softmax_scale
        scores = tl.where(row_mask, scores, float("-inf"))

        row_max = tl.max(scores, axis=0)
        p = tl.exp(scores - row_max)
        p = tl.where(row_mask, p, 0.0)
        denom_sm = tl.maximum(tl.sum(p, axis=0), 1e-6)
        p = p / denom_sm  # this row's softmax; exists only for this iteration

        S = S + p  # p is already 0 outside row_mask, so invalid rows are a no-op

        denom = tl.maximum(tl.minimum(i + 1, n_valid), 1)
        summary_row = S / denom

        rand_off = (pid * BLOCK_L + i) * BLOCK_L + offs_l
        rnd = tl.rand(seed, rand_off)
        keep = rnd >= dropout_p
        d_row = tl.where(keep & row_mask, p * (1.0 / (1.0 - dropout_p)), 0.0)

        P_i = tl.sum(d_row[:, None] * Vb, axis=0)

        tl.store(pb + i * stride_pl + offs_d * stride_pd, P_i, mask=mask_d & i_in_range)
        tl.store(sb + i * stride_si + offs_l * stride_sj, summary_row, mask=mask_l & i_in_range)
        # `p`, `scores`, `d_row` die here -- nothing L x L survives the iteration


@triton.jit
def _local_bwd_kernel(
    Q, K, V, dP, dSummary,
    dQ, dK, dV,
    stride_qb, stride_qh, stride_qg, stride_ql, stride_qd,
    stride_kb, stride_kh, stride_kg, stride_kl, stride_kd,
    stride_vb, stride_vh, stride_vg, stride_vl, stride_vd,
    stride_pb, stride_ph, stride_pg, stride_pl, stride_pd,
    stride_sb, stride_sh, stride_sg, stride_si, stride_sj,
    Token, L, G, H,
    softmax_scale, dropout_p, seed,
    HEAD_DIM: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_L: tl.constexpr,
):
    pid = tl.program_id(0)
    g = pid % G
    t = pid // G
    h = t % H
    b = t // H

    n_valid = tl.maximum(tl.minimum(L, Token - g * L), 0)

    offs_l = tl.arange(0, BLOCK_L)
    offs_d = tl.arange(0, BLOCK_D)
    mask_l = offs_l < L
    mask_d = offs_d < HEAD_DIM
    kv_mask = mask_l[:, None] & mask_d[None, :]

    qb = Q + b * stride_qb + h * stride_qh + g * stride_qg
    kb = K + b * stride_kb + h * stride_kh + g * stride_kg
    vb = V + b * stride_vb + h * stride_vh + g * stride_vg
    dpb = dP + b * stride_pb + h * stride_ph + g * stride_pg
    dsb = dSummary + b * stride_sb + h * stride_sh + g * stride_sg
    dqb = dQ + b * stride_qb + h * stride_qh + g * stride_qg

    Kb = tl.load(kb + offs_l[:, None] * stride_kl + offs_d[None, :] * stride_kd,
                 mask=kv_mask, other=0.0).to(tl.float32)
    Vb = tl.load(vb + offs_l[:, None] * stride_vl + offs_d[None, :] * stride_vd,
                 mask=kv_mask, other=0.0).to(tl.float32)

    R = tl.zeros((BLOCK_L,), dtype=tl.float32)          # suffix-sum accumulator, mirrors S
    dK_acc = tl.zeros((BLOCK_L, BLOCK_D), dtype=tl.float32)
    dV_acc = tl.zeros((BLOCK_L, BLOCK_D), dtype=tl.float32)

    for ridx in range(BLOCK_L):
        q = BLOCK_L - 1 - ridx  # walk rows in reverse: q = L-1 .. 0
        q_in_range = q < L
        q_valid = q < n_valid

        denom = tl.maximum(tl.minimum(q + 1, n_valid), 1)
        dS_row = tl.load(dsb + q * stride_si + offs_l * stride_sj,
                          mask=mask_l & q_in_range, other=0.0).to(tl.float32)
        R = R + dS_row / denom  # R now equals sum_{k=q}^{L-1} dSummary_k / denom_k

        row_mask = (offs_l <= q) & mask_l & q_valid

        Qq = tl.load(qb + q * stride_ql + offs_d * stride_qd,
                     mask=mask_d & q_in_range, other=0.0).to(tl.float32)

        # recompute row q's forward softmax -- nothing was saved to reuse
        scores = tl.sum(Qq[None, :] * Kb, axis=1) * softmax_scale
        scores = tl.where(row_mask, scores, float("-inf"))
        row_max = tl.max(scores, axis=0)
        p = tl.exp(scores - row_max)
        p = tl.where(row_mask, p, 0.0)
        denom_sm = tl.maximum(tl.sum(p, axis=0), 1e-6)
        p = p / denom_sm

        rand_off = (pid * BLOCK_L + q) * BLOCK_L + offs_l
        rnd = tl.rand(seed, rand_off)
        keep = rnd >= dropout_p
        d_row = tl.where(keep & row_mask, p * (1.0 / (1.0 - dropout_p)), 0.0)

        dPq = tl.load(dpb + q * stride_pl + offs_d * stride_pd,
                      mask=mask_d & q_in_range, other=0.0).to(tl.float32)

        # dL/dp via the P = dropout(p) @ V path
        g_from_p = tl.sum(Vb * dPq[None, :], axis=1)
        g_from_p = tl.where(keep & row_mask, g_from_p * (1.0 / (1.0 - dropout_p)), 0.0)
        # + dL/dp via the summary_causal = S / denom path (already suffix-summed into R)
        g_row = g_from_p + tl.where(row_mask, R, 0.0)

        # softmax backward: ds = p * (g - <p, g>)
        dot = tl.sum(p * g_row, axis=0)
        ds_row = tl.where(row_mask, p * (g_row - dot), 0.0)

        dQq = tl.sum(ds_row[:, None] * Kb, axis=0) * softmax_scale
        tl.store(dqb + q * stride_ql + offs_d * stride_qd, dQq, mask=mask_d & q_in_range)

        dK_acc += ds_row[:, None] * Qq[None, :] * softmax_scale
        dV_acc += d_row[:, None] * dPq[None, :]

    dkb = dK + b * stride_kb + h * stride_kh + g * stride_kg
    dvb = dV + b * stride_vb + h * stride_vh + g * stride_vg
    tl.store(dkb + offs_l[:, None] * stride_kl + offs_d[None, :] * stride_kd, dK_acc, mask=kv_mask)
    tl.store(dvb + offs_l[:, None] * stride_vl + offs_d[None, :] * stride_vd, dV_acc, mask=kv_mask)


class _LocalCausalAttentionFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, Token, dropout_p, training):
        B, H, G, L, d = Q.shape
        Qc, Kc, Vc = Q.contiguous(), K.contiguous(), V.contiguous()

        P = torch.empty_like(Qc)
        Summary = torch.empty(B, H, G, L, L, device=Q.device, dtype=Q.dtype)

        eff_dropout = float(dropout_p) if training else 0.0
        seed = random.randint(0, 2 ** 31 - 1) if eff_dropout > 0 else 0

        BLOCK_L = triton.next_power_of_2(L)
        BLOCK_D = triton.next_power_of_2(d)
        grid = (B * H * G,)

        _local_fwd_kernel[grid](
            Qc, Kc, Vc, P, Summary,
            Qc.stride(0), Qc.stride(1), Qc.stride(2), Qc.stride(3), Qc.stride(4),
            Kc.stride(0), Kc.stride(1), Kc.stride(2), Kc.stride(3), Kc.stride(4),
            Vc.stride(0), Vc.stride(1), Vc.stride(2), Vc.stride(3), Vc.stride(4),
            P.stride(0), P.stride(1), P.stride(2), P.stride(3), P.stride(4),
            Summary.stride(0), Summary.stride(1), Summary.stride(2), Summary.stride(3), Summary.stride(4),
            Token, L, G, H,
            1.0 / math.sqrt(d), eff_dropout, seed,
            HEAD_DIM=d, BLOCK_D=BLOCK_D, BLOCK_L=BLOCK_L,
        )

        ctx.save_for_backward(Qc, Kc, Vc)
        ctx.Token, ctx.L, ctx.G, ctx.H, ctx.d = Token, L, G, H, d
        ctx.eff_dropout, ctx.seed = eff_dropout, seed
        return P, Summary

    @staticmethod
    def backward(ctx, dP, dSummary):
        Qc, Kc, Vc = ctx.saved_tensors
        B = Qc.shape[0]
        L, G, H, d = ctx.L, ctx.G, ctx.H, ctx.d

        dPc = dP.contiguous()
        dSc = dSummary.contiguous()
        dQ = torch.zeros_like(Qc)
        dK = torch.zeros_like(Kc)
        dV = torch.zeros_like(Vc)

        BLOCK_L = triton.next_power_of_2(L)
        BLOCK_D = triton.next_power_of_2(d)
        grid = (B * H * G,)

        _local_bwd_kernel[grid](
            Qc, Kc, Vc, dPc, dSc,
            dQ, dK, dV,
            Qc.stride(0), Qc.stride(1), Qc.stride(2), Qc.stride(3), Qc.stride(4),
            Kc.stride(0), Kc.stride(1), Kc.stride(2), Kc.stride(3), Kc.stride(4),
            Vc.stride(0), Vc.stride(1), Vc.stride(2), Vc.stride(3), Vc.stride(4),
            dPc.stride(0), dPc.stride(1), dPc.stride(2), dPc.stride(3), dPc.stride(4),
            dSc.stride(0), dSc.stride(1), dSc.stride(2), dSc.stride(3), dSc.stride(4),
            ctx.Token, L, G, H,
            1.0 / math.sqrt(d), ctx.eff_dropout, ctx.seed,
            HEAD_DIM=d, BLOCK_D=BLOCK_D, BLOCK_L=BLOCK_L,
        )
        return dQ, dK, dV, None, None, None


def local_causal_attention(Q, K, V, Token, dropout_p, training):
    """Streaming local-causal attention.

    Q, K, V: (B, H, G, L, d), already block-reshaped and padded.
    Returns (P, summary_causal) exactly as the old vectorized code did,
    but the (B, H, G, L, L) probability matrix is never materialized --
    only P (B,H,G,L,d) and summary_causal (B,H,G,L,L) itself exist.
    """
    return _LocalCausalAttentionFn.apply(Q, K, V, Token, dropout_p, training)


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

    The local pass itself went through a similar change: it used to build
    the (L, L) causal probability matrix once (`local`) and read it twice
    -- once to get `P = local @ V`, once via `local.cumsum` to get
    `summary_causal`. That matrix is now never materialized. A Triton
    kernel (`local_causal_attention`) walks one query row at a time,
    immediately folding each row into `P` and into a running (L,)
    accumulator that becomes `summary_causal`, then discards the row. The
    backward pass mirrors this: it recomputes each row's softmax (nothing
    was saved) while walking rows in reverse, maintaining a running
    suffix-sum of d(summary_causal)/dS in place of the forward pass's
    running sum. Compute is unchanged (still O(L^2) per block); what's
    gone is the O(L^2) memory footprint of `local` itself.
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
        # cross-block level: STRICT causal mask (j < i) -- a block never
        # attends to itself via the pooled summary, its own contribution
        # comes entirely from the exact local pass instead. (The old
        # ordinary-causal `causal_mask` buffer that fed the local pass is
        # gone -- that masking now lives inside the Triton kernel.)
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
        # output, exact and fully causal, streamed row-by-row by a Triton
        # kernel so the (L, L) probability matrix is never materialized ---
        P, summary_causal = local_causal_attention(
            Q, K, V, Token, self.attn_dropout.p, self.training
        )
        P = P.to(V.dtype)

        # per-block pooled pattern: the key side of the cross-block step,
        # one vector per block regardless of query offset (safe, since a
        # block being read this way is entirely in the past). This is just
        # the last (whole-block) offset of summary_causal.
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
        # read): summary is the pooling weight, same role `local` used to
        # play for the own-block output above. Its own dropout call,
        # independent of the dropout applied inside the Triton kernel for
        # `P`, right alongside the other weights about to multiply V.
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

        out_gate = self.global_scale.view(1, self.n_head, 1, 1, 1)
        y = P + out_gate * cross

        y = y.permute(0, 2, 3, 1, 4).reshape(Batch, G * L, Embedding)
        if padding:
            y = y[:, :Token]
        y = self.c_proj(y)
        y = self.resid_dropout(y)
        return y