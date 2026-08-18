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
#   4. writes that row's contribution straight into `q_cross[i]` --
#      the pooled-pattern half from `S`, the content half from Q[i]
#      scaled by the (per-head) content gate -- and throws the row away.
#
# The only state carried between rows is `S` (shape (L,)) plus the K/V
# block loaded once at the top -- the (L, L) matrix of probabilities
# never exists as a tensor, on-chip or in HBM, and neither does the old
# (L, L) `summary_causal` tensor: its only consumer, `q_cross`, is
# written directly, so `summary_causal` is never materialized either.
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
# -- the mirror image of the forward pass's running `S`. It also carries
# the content-gate's own scalar accumulator `dCS_acc`, since folding
# `content_scale * Q` into the kernel moves that gradient inside too
# (it used to be a plain PyTorch multiply that autograd handled for
# free); dCS_acc is written out per (batch, head, block) program and
# summed over batch/block in Python afterward.
# =====================================================================


@triton.jit
def _local_fwd_kernel(
    Q, K, V, P, QCross, ContentScale,
    stride_qb, stride_qh, stride_qg, stride_ql, stride_qd,
    stride_kb, stride_kh, stride_kg, stride_kl, stride_kd,
    stride_vb, stride_vh, stride_vg, stride_vl, stride_vd,
    stride_pb, stride_ph, stride_pg, stride_pl, stride_pd,
    stride_qcb, stride_qch, stride_qcg, stride_qci, stride_qcj,
    Token, L, G, H,
    softmax_scale, dropout_p, seed, l_scale, d_scale,
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
    qcb = QCross + b * stride_qcb + h * stride_qch + g * stride_qcg

    content_scale = tl.load(ContentScale + h).to(tl.float32)

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

        # Write straight into q_cross = cat([summary_causal/L**.25, gate*Q/d**.25], -1)
        # instead of into a standalone (B,H,G,L,L) summary_causal buffer -- this row's
        # `summary_row` and `q_i` are already in registers, so no separate tensor is
        # ever allocated for summary_causal; only the (already-required) q_cross is.
        qc_row = qcb + i * stride_qci
        tl.store(qc_row + offs_l * stride_qcj, summary_row * l_scale,
                  mask=mask_l & i_in_range)
        tl.store(qc_row + (L + offs_d) * stride_qcj, content_scale * q_i * d_scale,
                  mask=mask_d & i_in_range)
        # `p`, `scores`, `d_row` die here -- nothing L x L survives the iteration


@triton.jit
def _local_bwd_kernel(
    Q, K, V, dP, dQCross, ContentScale,
    dQ, dK, dV, DContentScale,
    stride_qb, stride_qh, stride_qg, stride_ql, stride_qd,
    stride_kb, stride_kh, stride_kg, stride_kl, stride_kd,
    stride_vb, stride_vh, stride_vg, stride_vl, stride_vd,
    stride_pb, stride_ph, stride_pg, stride_pl, stride_pd,
    stride_qcb, stride_qch, stride_qcg, stride_qci, stride_qcj,
    stride_dcsb, stride_dcsh, stride_dcsg,
    Token, L, G, H,
    softmax_scale, dropout_p, seed, l_scale, d_scale,
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
    dqcb = dQCross + b * stride_qcb + h * stride_qch + g * stride_qcg
    dqb = dQ + b * stride_qb + h * stride_qh + g * stride_qg

    content_scale = tl.load(ContentScale + h).to(tl.float32)

    Kb = tl.load(kb + offs_l[:, None] * stride_kl + offs_d[None, :] * stride_kd,
                 mask=kv_mask, other=0.0).to(tl.float32)
    Vb = tl.load(vb + offs_l[:, None] * stride_vl + offs_d[None, :] * stride_vd,
                 mask=kv_mask, other=0.0).to(tl.float32)

    R = tl.zeros((BLOCK_L,), dtype=tl.float32)          # suffix-sum accumulator, mirrors S
    dK_acc = tl.zeros((BLOCK_L, BLOCK_D), dtype=tl.float32)
    dV_acc = tl.zeros((BLOCK_L, BLOCK_D), dtype=tl.float32)
    dCS_acc = tl.zeros((), dtype=tl.float32)            # content_scale grad, mirrors S too

    for ridx in range(BLOCK_L):
        q = BLOCK_L - 1 - ridx  # walk rows in reverse: q = L-1 .. 0
        q_in_range = q < L
        q_valid = q < n_valid

        denom = tl.maximum(tl.minimum(q + 1, n_valid), 1)

        # dQCross carries both halves of the row this time: the pooled-pattern
        # half (what used to be `dSummary`, read back out at its forward scale)
        # and the content half (Q's direct contribution to q_cross).
        dqc_row = dqcb + q * stride_qci
        dSummary_row = tl.load(dqc_row + offs_l * stride_qcj,
                                mask=mask_l & q_in_range, other=0.0).to(tl.float32) * l_scale
        dQCross_d = tl.load(dqc_row + (L + offs_d) * stride_qcj,
                             mask=mask_d & q_in_range, other=0.0).to(tl.float32)

        R = R + dSummary_row / denom  # R now equals sum_{k=q}^{L-1} dSummary_k / denom_k

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

        # dQ gets two contributions now: the usual one through the local softmax,
        # plus a new direct one from q_cross's content half (Q used to reach
        # q_cross only through a plain PyTorch multiply outside the kernel, where
        # autograd handled it for free; now that multiply happens in-kernel, its
        # gradient has to be accumulated here explicitly).
        dQq_local = tl.sum(ds_row[:, None] * Kb, axis=0) * softmax_scale
        dQq_cross = dQCross_d * content_scale * d_scale
        dQq = dQq_local + dQq_cross
        tl.store(dqb + q * stride_ql + offs_d * stride_qd, dQq, mask=mask_d & q_in_range)

        # d(content_scale) accumulates across rows the same way dK/dV do; the
        # per-(batch,head,block) partial is summed over batch/block in Python.
        dCS_acc += tl.sum(dQCross_d * Qq, axis=0) * d_scale

        dK_acc += ds_row[:, None] * Qq[None, :] * softmax_scale
        dV_acc += d_row[:, None] * dPq[None, :]

    dkb = dK + b * stride_kb + h * stride_kh + g * stride_kg
    dvb = dV + b * stride_vb + h * stride_vh + g * stride_vg
    tl.store(dkb + offs_l[:, None] * stride_kl + offs_d[None, :] * stride_kd, dK_acc, mask=kv_mask)
    tl.store(dvb + offs_l[:, None] * stride_vl + offs_d[None, :] * stride_vd, dV_acc, mask=kv_mask)

    dcs_ptr = DContentScale + b * stride_dcsb + h * stride_dcsh + g * stride_dcsg
    tl.store(dcs_ptr, dCS_acc)


class _LocalCausalAttentionFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, Token, dropout_p, training, content_scale):
        B, H, G, L, d = Q.shape
        Qc, Kc, Vc = Q.contiguous(), K.contiguous(), V.contiguous()
        cs = content_scale.contiguous().to(torch.float32)

        P = torch.empty_like(Qc)
        # q_cross = cat([summary_causal / L**.25, content_scale*Q / d**.25], -1),
        # written directly by the kernel -- summary_causal itself (B,H,G,L,L) is
        # never allocated, only this (B,H,G,L,L+d) buffer, which the caller needed
        # to build anyway.
        QCross = torch.empty(B, H, G, L, L + d, device=Q.device, dtype=Q.dtype)

        eff_dropout = float(dropout_p) if training else 0.0
        seed = random.randint(0, 2 ** 31 - 1) if eff_dropout > 0 else 0

        l_scale = float(L) ** -0.25
        d_scale = float(d) ** -0.25

        BLOCK_L = triton.next_power_of_2(L)
        BLOCK_D = triton.next_power_of_2(d)
        grid = (B * H * G,)

        _local_fwd_kernel[grid](
            Qc, Kc, Vc, P, QCross, cs,
            Qc.stride(0), Qc.stride(1), Qc.stride(2), Qc.stride(3), Qc.stride(4),
            Kc.stride(0), Kc.stride(1), Kc.stride(2), Kc.stride(3), Kc.stride(4),
            Vc.stride(0), Vc.stride(1), Vc.stride(2), Vc.stride(3), Vc.stride(4),
            P.stride(0), P.stride(1), P.stride(2), P.stride(3), P.stride(4),
            QCross.stride(0), QCross.stride(1), QCross.stride(2), QCross.stride(3), QCross.stride(4),
            Token, L, G, H,
            1.0 / math.sqrt(d), eff_dropout, seed, l_scale, d_scale,
            HEAD_DIM=d, BLOCK_D=BLOCK_D, BLOCK_L=BLOCK_L,
        )

        ctx.save_for_backward(Qc, Kc, Vc, cs)
        ctx.Token, ctx.L, ctx.G, ctx.H, ctx.d = Token, L, G, H, d
        ctx.eff_dropout, ctx.seed = eff_dropout, seed
        ctx.l_scale, ctx.d_scale = l_scale, d_scale
        ctx.content_scale_dtype = content_scale.dtype
        return P, QCross

    @staticmethod
    def backward(ctx, dP, dQCross):
        Qc, Kc, Vc, cs = ctx.saved_tensors
        B = Qc.shape[0]
        L, G, H, d = ctx.L, ctx.G, ctx.H, ctx.d

        dPc = dP.contiguous()
        dQCc = dQCross.contiguous()
        dQ = torch.zeros_like(Qc)
        dK = torch.zeros_like(Kc)
        dV = torch.zeros_like(Vc)
        # per-(batch, head, block) partial content_scale grad; reduced over
        # batch/block in Python below since content_scale is per-head only.
        dCS_partial = torch.zeros(B, H, G, device=Qc.device, dtype=torch.float32)

        BLOCK_L = triton.next_power_of_2(L)
        BLOCK_D = triton.next_power_of_2(d)
        grid = (B * H * G,)

        _local_bwd_kernel[grid](
            Qc, Kc, Vc, dPc, dQCc, cs,
            dQ, dK, dV, dCS_partial,
            Qc.stride(0), Qc.stride(1), Qc.stride(2), Qc.stride(3), Qc.stride(4),
            Kc.stride(0), Kc.stride(1), Kc.stride(2), Kc.stride(3), Kc.stride(4),
            Vc.stride(0), Vc.stride(1), Vc.stride(2), Vc.stride(3), Vc.stride(4),
            dPc.stride(0), dPc.stride(1), dPc.stride(2), dPc.stride(3), dPc.stride(4),
            dQCc.stride(0), dQCc.stride(1), dQCc.stride(2), dQCc.stride(3), dQCc.stride(4),
            dCS_partial.stride(0), dCS_partial.stride(1), dCS_partial.stride(2),
            ctx.Token, L, G, H,
            1.0 / math.sqrt(d), ctx.eff_dropout, ctx.seed, ctx.l_scale, ctx.d_scale,
            HEAD_DIM=d, BLOCK_D=BLOCK_D, BLOCK_L=BLOCK_L,
        )

        dContentScale = dCS_partial.sum(dim=(0, 2)).to(ctx.content_scale_dtype)
        return dQ, dK, dV, None, None, None, dContentScale


def local_causal_attention(Q, K, V, Token, dropout_p, training, content_scale):
    """Streaming local-causal attention, fused with the cross-block query build.

    Q, K, V: (B, H, G, L, d), already block-reshaped and padded.
    content_scale: (H,) per-head gate on the content half of the cross-block query.

    Returns (P, q_cross):
      - P: (B, H, G, L, d), the exact local-causal output (unchanged from before).
      - q_cross: (B, H, G, L, L + d), i.e.
            cat([summary_causal / L**0.25, content_scale * Q / d**0.25], dim=-1)
        computed and written a row at a time inside the same kernel that produces
        P. The old intermediate, `summary_causal` (B, H, G, L, L), never exists as
        a tensor -- q_cross was always the only thing it was used to build, so the
        kernel now produces q_cross's two halves directly instead of materializing
        summary_causal and concatenating in Python afterward.
    """
    return _LocalCausalAttentionFn.apply(Q, K, V, Token, dropout_p, training, content_scale)


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

    The local pass itself went through two rounds of the same kind of
    change. First, the (L, L) causal probability matrix that used to be
    built once (`local`) and read twice -- once for `P = local @ V`, once
    via `local.cumsum` for `summary_causal` -- was replaced by a Triton
    kernel (`local_causal_attention`) that walks one query row at a time,
    folding each row directly into `P` and into a running (L,) accumulator
    that becomes `summary_causal`, then discarding the row: compute
    unchanged (still O(L^2) per block), but the O(L^2) memory footprint of
    `local` itself gone. Second, `summary_causal`'s only remaining
    consumer -- concatenating it with a scaled, gated `Q` into `q_cross`
    for the cross-block step -- was folded into that same kernel too: each
    row already has `summary_row` and `Q_i` in registers when it's
    computed, so the kernel writes both halves of `q_cross` straight to
    HBM instead of writing `summary_causal` and letting Python's `torch.cat`
    read it back out. `summary_causal` (B,H,G,L,L) now never exists as a
    tensor at all -- only `q_cross` (B,H,G,L,L+d), which was already
    required downstream. Since `content_scale` now multiplies `Q` inside
    the kernel instead of in a plain PyTorch op, its gradient (and the new
    direct dQ contribution from `q_cross`'s content half) is accumulated by
    hand in the backward kernel, mirroring how dK/dV are already
    accumulated there.
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
        # kernel so neither the (L, L) probability matrix nor the (L, L)
        # summary_causal tensor is ever materialized -- the kernel emits
        # `P` and `q_cross` (summary_causal's only consumer) directly ---
        P, q_cross = local_causal_attention(
            Q, K, V, Token, self.attn_dropout.p, self.training, self.content_scale
        )
        P = P.to(V.dtype)

        # q_cross's own first L columns are exactly the old
        # `summary_causal / L**0.25`, so the whole-block pooling weights this
        # module still needs (for Kbar and the cross-block key) come straight
        # out of q_cross's last row -- no separate summary_causal tensor to
        # slice.
        summary_scaled = q_cross[..., -1, :L]  # == summary_causal[..., -1, :] / L**0.25
        summary = summary_scaled * (L ** 0.25)  # unscaled weights, for the content pool below

        # content-based routing key for the cross-block step: pool K the
        # same way `summary` pools local-attention mass, so the coarse
        # routing score can compare real content, not just attention shape.
        Kbar = torch.einsum('bhjk,bhjkd->bhjd', summary, K)  # (B, H, G, d)

        # --- combined pattern+content query/key for the cross-block step ---
        # k_block's first half needs exactly `summary / L**0.25`, i.e.
        # `summary_scaled` above -- already computed, just reuse it.
        k_block = torch.cat(
            [summary_scaled, Kbar / (head_dim ** 0.25)], dim=-1
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