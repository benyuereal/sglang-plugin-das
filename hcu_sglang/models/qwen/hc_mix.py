import torch
import triton
import triton.language as tl

from sglang.srt.environ import envs
from hcu_sglang.utils.detection import get_gcn_arch


@triton.jit
def _down_partial(X, W, P, M: tl.constexpr, K: tl.constexpr, R: tl.constexpr,
                  SPLITS: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    split = tl.program_id(1)
    m = tl.arange(0, 16)
    k = split * triton.cdiv(K, SPLITS * BK) * BK + tl.arange(0, BK)
    acc = tl.zeros((16, BN), tl.float32)
    for chunk in range(triton.cdiv(K, SPLITS * BK)):
        kk = k + chunk * BK
        x = tl.load(X + m[:, None] * K + kk[None, :],
                    (m[:, None] < M) & (kk[None, :] < K), 0)
        w = tl.load(W + n[:, None] * K + kk[None, :],
                    (n[:, None] < R) & (kk[None, :] < K), 0)
        acc = tl.dot(x, tl.trans(w), acc)
    tl.store(P + split * 16 * R + m[:, None] * R + n[None, :], acc,
             n[None, :] < R)


@triton.jit
def _reduce_silu(P, T, R: tl.constexpr, HC: tl.constexpr, SPLITS: tl.constexpr,
                 B: tl.constexpr):
    idx = tl.program_id(0) * B + tl.arange(0, B)
    s = tl.arange(0, SPLITS)
    partial = tl.load(P + s[:, None] * 16 * R + idx[None, :],
                      (s[:, None] < SPLITS) & (idx[None, :] < 16 * R), 0)
    value = tl.sum(partial, 0) / HC
    tl.store(T + idx, value * tl.sigmoid(value), idx < 16 * R)


@triton.jit
def _up_gate(X, T, W, Y, M: tl.constexpr, R: tl.constexpr, HC: tl.constexpr,
             HS: tl.constexpr, BJ: tl.constexpr, BR: tl.constexpr):
    m = tl.arange(0, 16)
    j = tl.program_id(0) * BJ + tl.arange(0, BJ)
    g = tl.arange(0, HC)
    n = tl.reshape(g[:, None] * HS + j[None, :], (HC * BJ,))
    nr = tl.reshape(tl.broadcast_to(j[None, :] < HS, (HC, BJ)), (HC * BJ,))
    r = tl.arange(0, BR)
    acc = tl.zeros((16, HC * BJ), tl.float32)
    for i in range(triton.cdiv(R, BR)):
        rr = i * BR + r
        t = tl.load(T + m[:, None] * R + rr[None, :], rr[None, :] < R, 0)
        w = tl.load(W + n[:, None] * R + rr[None, :],
                    nr[:, None] & (rr[None, :] < R), 0)
        acc = tl.dot(t, tl.trans(w), acc)
    gate = tl.sigmoid(tl.reshape(acc, (16, HC, BJ)))
    x = tl.load(X + m[:, None, None] * HC * HS + g[None, :, None] * HS + j[None, None, :],
                (m[:, None, None] < M) & (j[None, None, :] < HS), 0).to(tl.float32)
    y = tl.sum(gate * x, 1) / HC
    tl.store(Y + m[:, None] * HS + j[None, :], y,
             (m[:, None] < M) & (j[None, :] < HS))


def split_mix(x, down, up, hc, hs, splits=32, bn=32, bj=32):
    m, k = x.shape
    rank = down.shape[0]
    partial = torch.empty((splits, 16, rank), device=x.device, dtype=torch.float32)
    intermediate = torch.empty((16, rank), device=x.device, dtype=x.dtype)
    output = torch.empty((m, hs), device=x.device, dtype=x.dtype)
    _down_partial[(triton.cdiv(rank, bn), splits)](
        x, down, partial, m, k, rank, splits, bn, 512, num_warps=4)
    _reduce_silu[(triton.cdiv(16 * rank, 256),)](
        partial, intermediate, rank, hc, splits, 256, num_warps=4)
    _up_gate[(triton.cdiv(hs, bj),)](
        x, intermediate, up, output, m, rank, hc, hs, bj, 64, num_warps=4)
    return output


def around_fused_hc_mix(original, hyper_input_normed, w_down, w_up, hc, hs):
    if not envs.SGLANG_QWEN4_SPLIT_HC_MIX.get():
        return original(hyper_input_normed, w_down, w_up, hc, hs)
    x = hyper_input_normed
    if not (
        x.is_cuda and x.dtype == torch.bfloat16 and x.dim() == 2
        and 0 < x.shape[0] <= 16 and x.shape[1] == 10240
        and hc == 4 and hs == 2560 and get_gcn_arch().startswith("gfx936")
        and w_down.shape == (320, 10240) and w_up.shape == (10240, 320)
        and w_down.dtype == w_up.dtype == x.dtype
        and w_down.device == w_up.device == x.device
        and x.is_contiguous() and w_down.is_contiguous() and w_up.is_contiguous()
    ):
        return original(hyper_input_normed, w_down, w_up, hc, hs)
    return split_mix(x, w_down, w_up, hc, hs)
