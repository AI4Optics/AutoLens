import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as nnF
import triton
import triton.language as tl

from .utils import *

EPSILON: tl.constexpr = epsilon
BKV_LIST = [
    512,
    1024,
    2048,
    # 16,
]


@triton.autotune(
    configs=[
        triton.Config({"BS": BS}, num_warps=num_warps, num_stages=num_stages)
        for BS in BKV_LIST
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
        # for num_warps in [2]
        # for num_stages in [2]
    ],
    key=["T"],
)
@triton.jit
def flash_sag_fwd_kernel(
    i_x_ptr,  # (T)
    i_y_ptr,  # (T)
    o_z_ptr,  # (T)
    k_ptr,  # (1)
    c_ptr,  # (1)
    ai2_ptr,  # (1)
    ai4_ptr,  # (1)
    ai6_ptr,  # (1)
    ai8_ptr,  # (1)
    ai10_ptr,  # (1)
    ai12_ptr,  # (1)
    T,
    BS: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BS + tl.arange(0, BS)
    mask = offsets < T
    B_o0 = tl.load(i_x_ptr + offsets, mask=mask)
    B_o1 = tl.load(i_y_ptr + offsets, mask=mask)

    k, c = tl.load(k_ptr), tl.load(c_ptr)
    ai2, ai4, ai6, ai8, ai10, ai12 = tl.load(ai2_ptr), tl.load(ai4_ptr), tl.load(ai6_ptr), tl.load(ai8_ptr), tl.load(ai10_ptr), tl.load(ai12_ptr)

    r2 = B_o0 * B_o0 + B_o1 * B_o1
    if k > -1:
        valid = r2 < (1 / (c * c) / (1 + k))
    else:
        valid = (tl.zeros_like(B_o0) + 1).to(tl.int1)

    valid_x, valid_y = B_o0 * valid, B_o1 * valid
    valid_r2 = valid_x * valid_x + valid_y * valid_y

    valid_r2_2 = valid_r2 * valid_r2
    valid_r2_3, valid_r2_4 = valid_r2_2 * valid_r2, valid_r2_2 * valid_r2_2
    valid_r2_5, valid_r2_6 = valid_r2_3 * valid_r2_2, valid_r2_3 * valid_r2_3
    sf = tl.sqrt(1 - (1 + k) * valid_r2 * c * c + EPSILON)
    ts = valid_r2 * c / (1 + sf)
    ts1 = ts + ai2 * valid_r2 + ai4 * valid_r2_2 + ai6 * valid_r2_3 + ai8 * valid_r2_4 + ai10 * valid_r2_5 + ai12 * valid_r2_6
    tl.store(o_z_ptr + offsets, ts1, mask=mask)


def flash_sag_fwd(i_x, i_y, k, c, ai2, ai4, ai6, ai8, ai10, ai12):


    input_shape = i_x.shape
    i_x, i_y = i_x.reshape(-1), i_y.reshape(-1)
    (T,) = i_x.shape
    o_z = torch.zeros_like(i_x)
    grid = lambda meta: (triton.cdiv(T, meta["BS"]),)
    flash_sag_fwd_kernel[grid](i_x, i_y, o_z, k, c, ai2, ai4, ai6, ai8, ai10, ai12, T)
    o_z = o_z.reshape(input_shape)
    return o_z




@triton.jit
def flash_sag_bwd_kernel(
    # fmt: off
    i_x_ptr, i_y_ptr, i_dz_ptr , k_ptr,c_ptr,
    ai2_ptr,ai4_ptr,ai6_ptr,ai8_ptr, ai10_ptr, ai12_ptr,locks_ptr,
    o_dx_ptr, o_dy_ptr,o_c_ptr,o_ai2_ptr,o_ai4_ptr, o_ai6_ptr, o_ai8_ptr, o_ai10_ptr,o_ai12_ptr,
    T,BS: tl.constexpr
    # fmt: on
):

    pid = tl.program_id(0)
    LOCK = locks_ptr + pid
    offsets = pid * BS + tl.arange(0, BS)
    mask = offsets < T
    B_o0 = tl.load(i_x_ptr + offsets, mask=mask)
    B_o1 = tl.load(i_y_ptr + offsets, mask=mask)
    B_dz = tl.load(i_dz_ptr + offsets, mask=mask)

    k, c = tl.load(k_ptr), tl.load(c_ptr)
    ai2, ai4, ai6, ai8, ai10, ai12 = tl.load(ai2_ptr), tl.load(ai4_ptr), tl.load(ai6_ptr), tl.load(ai8_ptr), tl.load(ai10_ptr), tl.load(ai12_ptr)

    r2 = B_o0 * B_o0 + B_o1 * B_o1
    if k > -1:
        valid = r2 < (1 / (c * c) / (1 + k))
    else:
        valid = (tl.zeros_like(B_o0) + 1).to(tl.int1)

    valid_x, valid_y = B_o0 * valid, B_o1 * valid
    valid_r2 = valid_x * valid_x + valid_y * valid_y

    valid_r2_2 = valid_r2 * valid_r2
    valid_r2_3, valid_r2_4 = valid_r2_2 * valid_r2, valid_r2_2 * valid_r2_2
    valid_r2_5, valid_r2_6 = valid_r2_3 * valid_r2_2, valid_r2_3 * valid_r2_3
    sf = tl.sqrt(1 - (1 + k) * valid_r2 * c * c + EPSILON)
    ts = valid_r2 * c / (1 + sf)
    # ts1 = ts + ai2 * valid_r2 + ai4 * valid_r2_2 + ai6 * valid_r2_3 + ai8 * valid_r2_4 + ai10 * valid_r2_5 + ai12 * valid_r2_6

    dvalidx_o0, dvalidy_o1 = valid, valid
    dvalid_r2_o0, dvalid_r2_o1 = 2 * valid_x * dvalidx_o0, 2 * valid_y * dvalidy_o1
    psf_valid_r2 = -1 / 2 / sf * (1 + k) * c * c
    dsf_c = -1 / sf * (1 + k) * valid_r2 * c
    dsf_k = -1 / 2 / sf * valid_r2 * c * c
    dsf_o0, dsf_o1 = psf_valid_r2 * dvalid_r2_o0, psf_valid_r2 * dvalid_r2_o1
    pts_sf = -1 / (1 + sf) / (1 + sf) * valid_r2 * c
    pts_valid_r2 = c / (1 + sf)
    pts_c = valid_r2 / (1 + sf)
    dts_c = pts_c + pts_sf * dsf_c
    (dts_o0, dts_o1) = (pts_valid_r2 * dvalid_r2_o0 + pts_sf * dsf_o0, pts_valid_r2 * dvalid_r2_o1 + pts_sf * dsf_o1)
    pts_1_ts = 1
    pts_1_valid_r2 = ai2 + 2 * ai4 * valid_r2 + 3 * ai6 * valid_r2_2 + 4 * ai8 * valid_r2_4 + 5 * ai10 * valid_r2_4 + 6 * ai12 * valid_r2_5
    dz_c = B_dz * dts_c
    dts_1_o0 = pts_1_ts * dts_o0 + pts_1_valid_r2 * dvalid_r2_o0
    dts_1_o1 = pts_1_ts * dts_o1 + pts_1_valid_r2 * dvalid_r2_o1
    dz_o0 = B_dz * dts_1_o0
    dz_o1 = B_dz * dts_1_o1
    dz_ai2 = B_dz * valid_r2
    dz_ai4 = B_dz * valid_r2_2
    dz_ai6 = B_dz * valid_r2_3
    dz_ai8 = B_dz * valid_r2_4
    dz_ai10 = B_dz * valid_r2_5
    dz_ai12 = B_dz * valid_r2_6

    tl.store(o_dx_ptr + offsets, dz_o0, mask=mask)
    tl.store(o_dy_ptr + offsets,  dz_o1, mask=mask)

    dz_c = tl.where(mask, dz_c, 0)
    dz_ai2 = tl.where(mask, dz_ai2, 0)
    dz_ai4 = tl.where(mask, dz_ai4, 0)
    dz_ai6 = tl.where(mask, dz_ai6, 0)
    dz_ai8 = tl.where(mask, dz_ai8, 0)
    dz_ai10 = tl.where(mask, dz_ai10, 0)
    dz_ai12 = tl.where(mask, dz_ai12, 0)
    while tl.atomic_cas(LOCK, 0, 1) == 1:
        pass
    dz_c_sum = tl.sum(dz_c)
    dz_ai2_sum = tl.sum(dz_ai2)
    dz_ai4_sum = tl.sum(dz_ai4)
    dz_ai6_sum = tl.sum(dz_ai6)
    dz_ai8_sum = tl.sum(dz_ai8)
    dz_ai10_sum = tl.sum(dz_ai10)
    dz_ai12_sum = tl.sum(dz_ai12)
    tl.store(o_ai2_ptr + pid, dz_ai2_sum)
    tl.store(o_ai4_ptr + pid, dz_ai4_sum)
    tl.store(o_ai6_ptr + pid, dz_ai6_sum)
    tl.store(o_ai8_ptr + pid, dz_ai8_sum)
    tl.store(o_ai10_ptr + pid, dz_ai10_sum)
    tl.store(o_ai12_ptr + pid, dz_ai12_sum)
    tl.store(o_c_ptr + pid, dz_c_sum)
    tl.atomic_xchg(LOCK, 0)


def flash_sag_bwd(i_x, i_y, i_dz, k, c, ai2, ai4, ai6, ai8, ai10, ai12):

    input_shape = i_x.shape

    i_x, i_y = i_x.reshape(-1), i_y.reshape(-1)
    (T,) = i_x.shape
    o_dx, o_dy = torch.zeros_like(i_x), torch.zeros_like(i_y)
    BLOCK_SIZE = 512
    BN = triton.cdiv(T, BLOCK_SIZE)
    locks = torch.zeros((BN,), dtype=torch.int32, device=c.device)
    o_c = torch.zeros((BN,), device=c.device, dtype=c.dtype)
    o_ai2, o_ai4, o_ai6, o_ai8, o_ai10, o_ai12 = (
        torch.zeros_like(o_c),
        torch.zeros_like(o_c),
        torch.zeros_like(o_c),
        torch.zeros_like(o_c),
        torch.zeros_like(o_c),
        torch.zeros_like(o_c),
    )
    grid = (BN,)
    # fmt:off
    flash_sag_bwd_kernel[grid]( 
        i_x, i_y, i_dz, k, c, 
        ai2, ai4, ai6, ai8, ai10, ai12, locks,
        o_dx, o_dy, o_c, o_ai2, o_ai4,  o_ai6, o_ai8, o_ai10, o_ai12,
        T,BS=BLOCK_SIZE,
    )
    # fmt: on
    o_dx, o_dy = o_dx.reshape(input_shape), o_dy.reshape(input_shape)
    return o_dx, o_dy, o_c, o_ai2, o_ai4, o_ai6, o_ai8, o_ai10, o_ai12


class FlashSagMethodFunction(torch.autograd.Function):
    @staticmethod
    @contiguous
    @autocast_custom_fwd
    def forward(ctx, i_x, i_y, k, c, ai2, ai4, ai6, ai8, ai10, ai12):

        o_z = flash_sag_fwd(i_x, i_y, k, c, ai2, ai4, ai6, ai8, ai10, ai12)
        ctx.save_for_backward(i_x, i_y, k, c, ai2, ai4, ai6, ai8, ai10, ai12)
        return o_z

    @staticmethod
    @contiguous
    @autocast_custom_bwd
    def backward(ctx, dz):
        i_x, i_y, k, c, ai2, ai4, ai6, ai8, ai10, ai12 = ctx.saved_tensors
        o_dx, o_dy, o_c, o_ai2, o_ai4, o_ai6, o_ai8, o_ai10, o_ai12 = flash_sag_bwd(i_x, i_y, dz, k, c, ai2, ai4, ai6, ai8, ai10, ai12)

        dz_ai2, dz_ai4, dz_ai6, dz_ai8, dz_ai10, dz_ai12, dz_c = (
            torch.sum(o_ai2, 0, keepdim=True),
            torch.sum(o_ai4, 0, keepdim=True),
            torch.sum(o_ai6, 0, keepdim=True),
            torch.sum(o_ai8, 0, keepdim=True),
            torch.sum(o_ai10, 0, keepdim=True),
            torch.sum(o_ai12, 0, keepdim=True),
            torch.sum(o_c, 0, keepdim=True),
        )
        return o_dx, o_dy, None, dz_c, dz_ai2, dz_ai4, dz_ai6, dz_ai8, dz_ai10, dz_ai12


@torch.compiler.disable
def flash_sag(i_x, i_y, k, c, ai2, ai4, ai6, ai8, ai10, ai12):
    z = FlashSagMethodFunction.apply(i_x, i_y, k, c, ai2, ai4, ai6, ai8, ai10, ai12)
    return z
