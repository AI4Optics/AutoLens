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
def flash_newtons_method_fwd_kernel(
    i_o_ptr,  # (T,3)
    i_d_ptr,  # (T,3)
    i_ra_ptr,  # (T,1)
    o_o_ptr,  # (T,3)
    o_ra_ptr,
    o_prev_t_ptr,  # (T,1)
    # o_t_ptr,
    # o_valid_ptr,
    d_ptr,  # (1)
    k_ptr,  # (1)
    c_ptr,  # (1)
    r,
    ai2_ptr,  # (1)
    ai4_ptr,  # (1)
    ai6_ptr,  # (1)
    ai8_ptr,  # (1)
    ai10_ptr,  # (1)
    ai12_ptr,  # (1)
    tol_i,
    tol_t,
    max_iter,
    step_bound,
    T,
    D,
    BS: tl.constexpr,
):
    pid = tl.program_id(0)
    # data_shape, block_shape = (S, G1, G2, D), (BS, G1, G2, 1)
    # Block o0 ptr
    B_o0_ptr = tl.make_block_ptr(i_o_ptr, (T, D), (D, 1), (pid * BS, 0), (BS, 1), order=(1, 0))
    B_o1_ptr = tl.make_block_ptr(i_o_ptr, (T, D), (D, 1), (pid * BS, 1), (BS, 1), order=(1, 0))
    B_o2_ptr = tl.make_block_ptr(i_o_ptr, (T, D), (D, 1), (pid * BS, 2), (BS, 1), order=(1, 0))
    B_d0_ptr = tl.make_block_ptr(i_d_ptr, (T, D), (D, 1), (pid * BS, 0), (BS, 1), order=(1, 0))
    B_d1_ptr = tl.make_block_ptr(i_d_ptr, (T, D), (D, 1), (pid * BS, 1), (BS, 1), order=(1, 0))
    B_d2_ptr = tl.make_block_ptr(i_d_ptr, (T, D), (D, 1), (pid * BS, 2), (BS, 1), order=(1, 0))
    B_ra_ptr = tl.make_block_ptr(i_ra_ptr, (T, 1), (1, 1), (pid * BS, 0), (BS, 1), order=(1, 0))

    O_B_o0_ptr = tl.make_block_ptr(o_o_ptr, (T, D), (D, 1), (pid * BS, 0), (BS, 1), order=(1, 0))
    O_B_o1_ptr = tl.make_block_ptr(o_o_ptr, (T, D), (D, 1), (pid * BS, 1), (BS, 1), order=(1, 0))
    O_B_o2_ptr = tl.make_block_ptr(o_o_ptr, (T, D), (D, 1), (pid * BS, 2), (BS, 1), order=(1, 0))
    O_B_ra_ptr = tl.make_block_ptr(o_ra_ptr, (T, 1), (1, 1), (pid * BS, 0), (BS, 1), order=(1, 0))
    O_B_prev_t_ptr = tl.make_block_ptr(o_prev_t_ptr, (T, 1), (1, 1), (pid * BS, 0), (BS, 1), order=(1, 0))

    B_o0 = tl.load(B_o0_ptr, boundary_check=(0, 1), padding_option="zero")
    B_o1 = tl.load(B_o1_ptr, boundary_check=(0, 1), padding_option="zero")
    B_o2 = tl.load(B_o2_ptr, boundary_check=(0, 1), padding_option="zero")
    B_d0 = tl.load(B_d0_ptr, boundary_check=(0, 1), padding_option="zero")
    B_d1 = tl.load(B_d1_ptr, boundary_check=(0, 1), padding_option="zero")
    B_d2 = tl.load(B_d2_ptr, boundary_check=(0, 1), padding_option="zero")
    B_ra = tl.load(B_ra_ptr, boundary_check=(0, 1), padding_option="zero")
    d, k, c = tl.load(d_ptr), tl.load(k_ptr), tl.load(c_ptr)
    ai2, ai4, ai6, ai8, ai10, ai12 = tl.load(ai2_ptr), tl.load(ai4_ptr), tl.load(ai6_ptr), tl.load(ai8_ptr), tl.load(ai10_ptr), tl.load(ai12_ptr)
    t0 = (d - B_o2) / B_d2
    ft = tl.zeros_like(B_o0) + 1
    it = 0
    t = t0
    valid = (tl.zeros_like(B_o0) + 1).to(tl.int1)
    continue_iter = tl.sum(tl.abs((tl.abs(ft) > tol_i).to(tl.float32))) > 0
    while continue_iter and it < max_iter:
        it += 1
        new_o0 = B_o0 + t * B_d0
        new_o1 = B_o1 + t * B_d1
        new_o2 = B_o2 + t * B_d2
        r2 = new_o0 * new_o0 + new_o1 * new_o1
        if k > -1:
            valid = r2 < (1 / (c * c) / (1 + k))
        else:
            valid = (tl.zeros_like(B_o0) + 1).to(tl.int1)
        valid = valid & (B_ra > 0)
        # sag
        valid_x, valid_y = new_o0 * valid, new_o1 * valid
        valid_r2 = valid_x * valid_x + valid_y * valid_y
        sf = tl.sqrt(1 - (1 + k) * valid_r2 * c * c + EPSILON)
        ts = valid_r2 * c / (1 + sf)
        ts += (
            ai2 * valid_r2
            + ai4 * valid_r2 * valid_r2
            + ai6 * valid_r2 * valid_r2 * valid_r2
            + ai8 * valid_r2 * valid_r2 * valid_r2 * valid_r2
            + ai10 * valid_r2 * valid_r2 * valid_r2 * valid_r2 * valid_r2
            + ai12 * valid_r2 * valid_r2 * valid_r2 * valid_r2 * valid_r2 * valid_r2
        )
        ft = ts + d - new_o2

        dsdr2 = (1 + sf + (1 + k) * valid_r2 * c * c / 2 / sf) * c / (1 + sf) / (1 + sf)
        dsdr2 += (
            ai2
            + 2 * ai4 * valid_r2
            + 3 * ai6 * valid_r2 * valid_r2
            + 4 * ai8 * valid_r2 * valid_r2 * valid_r2
            + 5 * ai10 * valid_r2 * valid_r2 * valid_r2 * valid_r2
            + 6 * ai12 * valid_r2 * valid_r2 * valid_r2 * valid_r2 * valid_r2
        )
        dfdx, dfdy = 2 * dsdr2 * valid_x, 2 * dsdr2 * valid_y
        dfdz = tl.zeros_like(valid_x) - 1
        dfdt = dfdx * B_d0 + dfdy * B_d1 + dfdz * B_d2
        t = t - tl.clamp(ft / (dfdt + EPSILON), -step_bound, step_bound)
        continue_iter = tl.sum(tl.abs((tl.abs(ft) > tol_i).to(tl.float32))) > 0
    tl.store(O_B_prev_t_ptr, t, boundary_check=(0, 1))
    new_o0 = B_o0 + t * B_d0
    new_o1 = B_o1 + t * B_d1
    new_o2 = B_o2 + t * B_d2
    r2 = new_o0 * new_o0 + new_o1 * new_o1
    # sag
    if k > -1:
        valid = r2 < (1 / (c * c) / (1 + k))
    else:
        valid = (tl.zeros_like(B_o0) + 1).to(tl.int1)
    valid = valid & (B_ra > 0)
    # sag
    valid_x, valid_y = new_o0 * valid, new_o1 * valid
    valid_r2 = valid_x * valid_x + valid_y * valid_y
    sf = tl.sqrt(1 - (1 + k) * valid_r2 * c * c + EPSILON)
    ts = valid_r2 * c / (1 + sf)
    # total_surface = valid_r2 * c / (1 + tl.sqrt(1 - (1 + k) * valid_r2 * c * c) + EPSILON)
    ts += (
        ai2 * valid_r2
        + ai4 * valid_r2 * valid_r2
        + ai6 * valid_r2 * valid_r2 * valid_r2
        + ai8 * valid_r2 * valid_r2 * valid_r2 * valid_r2
        + ai10 * valid_r2 * valid_r2 * valid_r2 * valid_r2 * valid_r2
        + ai12 * valid_r2 * valid_r2 * valid_r2 * valid_r2 * valid_r2 * valid_r2
    )
    ft = ts + d - new_o2

    # sf = tl.sqrt(1 - (1 + k) * valid_r2 * c * c + EPSILON)
    dsdr2 = (1 + sf + (1 + k) * valid_r2 * c * c / 2 / sf) * c / (1 + sf) / (1 + sf)
    dsdr2 += (
        ai2
        + 2 * ai4 * valid_r2
        + 3 * ai6 * valid_r2 * valid_r2
        + 4 * ai8 * valid_r2 * valid_r2 * valid_r2
        + 5 * ai10 * valid_r2 * valid_r2 * valid_r2 * valid_r2
        + 6 * ai12 * valid_r2 * valid_r2 * valid_r2 * valid_r2 * valid_r2
    )
    dfdx, dfdy = 2 * dsdr2 * valid_x, 2 * dsdr2 * valid_y
    dfdz = tl.zeros_like(valid_x) - 1
    dfdt = dfdx * B_d0 + dfdy * B_d1 + dfdz * B_d2

    ft_dfdt = ft / (dfdt + EPSILON)
    final_t = t - tl.clamp(ft_dfdt, -step_bound, step_bound)

    fvalid = valid & (valid_r2 <= r * r)
    fvalid_x, fvalid_y = new_o0 * fvalid, new_o1 * fvalid
    fvalid_r2 = fvalid_x * fvalid_x + fvalid_y * fvalid_y
    fsf = tl.sqrt(1 - (1 + k) * fvalid_r2 * c * c + EPSILON)
    fts = fvalid_r2 * c / (1 + fsf)
    # total_surface = valid_r2 * c / (1 + tl.sqrt(1 - (1 + k) * valid_r2 * c * c) + EPSILON)
    fts += (
        ai2 * fvalid_r2
        + ai4 * fvalid_r2 * fvalid_r2
        + ai6 * fvalid_r2 * fvalid_r2 * fvalid_r2
        + ai8 * fvalid_r2 * fvalid_r2 * fvalid_r2 * fvalid_r2
        + ai10 * fvalid_r2 * fvalid_r2 * fvalid_r2 * fvalid_r2 * fvalid_r2
        + ai12 * fvalid_r2 * fvalid_r2 * fvalid_r2 * fvalid_r2 * fvalid_r2 * fvalid_r2
    )
    fft = fts + d - new_o2
    ffvalid = fvalid & (tl.abs(fft) < tol_t) & (final_t > 0)

    fnew_o0 = B_o0 + final_t * B_d0
    fnew_o1 = B_o1 + final_t * B_d1
    fnew_o2 = B_o2 + final_t * B_d2
    vfnew_o0 = tl.where(ffvalid, fnew_o0, B_o0)
    vfnew_o1 = tl.where(ffvalid, fnew_o1, B_o1)
    vfnew_o2 = tl.where(ffvalid, fnew_o2, B_o2)
    fnew_ra = B_ra * ffvalid

    tl.store(O_B_o0_ptr, vfnew_o0, boundary_check=(0, 1))
    tl.store(O_B_o1_ptr, vfnew_o1, boundary_check=(0, 1))
    tl.store(O_B_o2_ptr, vfnew_o2, boundary_check=(0, 1))
    tl.store(O_B_ra_ptr, fnew_ra, boundary_check=(0, 1))


def flash_newtons_method_fwd(i_o, i_d, i_ra, d, k, c, r, ai2, ai4, ai6, ai8, ai10, ai12, tol_i, tol_t, max_iter, step_bound):


    input_shape = i_o.shape
    i_o, i_d, i_ra = i_o.reshape(-1, input_shape[-1]), i_d.reshape(-1, input_shape[-1]), i_ra.reshape(-1, 1)
    T, D = i_o.shape
    o_o, o_ra = torch.zeros_like(i_o), torch.zeros_like(i_ra)
    o_prev_t = torch.zeros((T, 1), dtype=i_o.dtype, device=i_o.device)

    grid = lambda meta: (triton.cdiv(T, meta["BS"]),)
    flash_newtons_method_fwd_kernel[grid](
        i_o, i_d, i_ra, o_o, o_ra, o_prev_t, d, k, c, r, ai2, ai4, ai6, ai8, ai10, ai12, tol_i, tol_t, max_iter, step_bound, T, D
    )
    o_o, o_ra, o_prev_t = o_o.reshape(input_shape), o_ra.reshape(input_shape[:-1]), o_prev_t.reshape(input_shape[:-1])

    return o_o, o_ra, o_prev_t  # , o_t, o_valid


@triton.jit
def flash_newtons_method_bwd_kernel(
    i_o_ptr,
    i_d_ptr,
    i_ra_ptr,
    i_prev_t_ptr,
    i_do_ptr,
    i_dra_ptr,
    d_ptr,
    k_ptr,
    c_ptr,
    r,
    ai2_ptr,
    ai4_ptr,
    ai6_ptr,
    ai8_ptr,
    ai10_ptr,
    ai12_ptr,
    locks_ptr,
    o_do_ptr,
    o_dd_ptr,
    o_dra_ptr,
    o_d_ptr,
    o_k_ptr,
    o_c_ptr,
    o_ai2_ptr,
    o_ai4_ptr,
    o_ai6_ptr,
    o_ai8_ptr,
    o_ai10_ptr,
    o_ai12_ptr,
    tol_t,
    step_bound,
    T,
    D,
    BS: tl.constexpr,
):
    pid = tl.program_id(0)
    LOCK = locks_ptr + pid
    B_o0_ptr = tl.make_block_ptr(i_o_ptr, (T, D), (D, 1), (pid * BS, 0), (BS, 1), order=(1, 0))
    B_o1_ptr = tl.make_block_ptr(i_o_ptr, (T, D), (D, 1), (pid * BS, 1), (BS, 1), order=(1, 0))
    B_o2_ptr = tl.make_block_ptr(i_o_ptr, (T, D), (D, 1), (pid * BS, 2), (BS, 1), order=(1, 0))
    B_d0_ptr = tl.make_block_ptr(i_d_ptr, (T, D), (D, 1), (pid * BS, 0), (BS, 1), order=(1, 0))
    B_d1_ptr = tl.make_block_ptr(i_d_ptr, (T, D), (D, 1), (pid * BS, 1), (BS, 1), order=(1, 0))
    B_d2_ptr = tl.make_block_ptr(i_d_ptr, (T, D), (D, 1), (pid * BS, 2), (BS, 1), order=(1, 0))
    i_B_do0_ptr = tl.make_block_ptr(i_do_ptr, (T, D), (D, 1), (pid * BS, 0), (BS, 1), order=(1, 0))
    i_B_do1_ptr = tl.make_block_ptr(i_do_ptr, (T, D), (D, 1), (pid * BS, 1), (BS, 1), order=(1, 0))
    i_B_do2_ptr = tl.make_block_ptr(i_do_ptr, (T, D), (D, 1), (pid * BS, 2), (BS, 1), order=(1, 0))

    O_B_do0_ptr = tl.make_block_ptr(o_do_ptr, (T, D), (D, 1), (pid * BS, 0), (BS, 1), order=(1, 0))
    O_B_do1_ptr = tl.make_block_ptr(o_do_ptr, (T, D), (D, 1), (pid * BS, 1), (BS, 1), order=(1, 0))
    O_B_do2_ptr = tl.make_block_ptr(o_do_ptr, (T, D), (D, 1), (pid * BS, 2), (BS, 1), order=(1, 0))
    O_B_dd0_ptr = tl.make_block_ptr(o_dd_ptr, (T, D), (D, 1), (pid * BS, 0), (BS, 1), order=(1, 0))
    O_B_dd1_ptr = tl.make_block_ptr(o_dd_ptr, (T, D), (D, 1), (pid * BS, 1), (BS, 1), order=(1, 0))
    O_B_dd2_ptr = tl.make_block_ptr(o_dd_ptr, (T, D), (D, 1), (pid * BS, 2), (BS, 1), order=(1, 0))
    O_B_dra_ptr = tl.make_block_ptr(o_dra_ptr, (T, 1), (1, 1), (pid * BS, 0), (BS, 1), order=(1, 0))

    i_B_ra_ptr = tl.make_block_ptr(i_ra_ptr, (T, 1), (1, 1), (pid * BS, 0), (BS, 1), order=(1, 0))
    i_B_dra_ptr = tl.make_block_ptr(i_dra_ptr, (T, 1), (1, 1), (pid * BS, 0), (BS, 1), order=(1, 0))
    i_B_prev_t_ptr = tl.make_block_ptr(i_prev_t_ptr, (T, 1), (1, 1), (pid * BS, 0), (BS, 1), order=(1, 0))

    mask = (pid * BS + tl.arange(0, BS))[:, None] < T
    B_o0 = tl.load(B_o0_ptr, boundary_check=(0, 1), padding_option="zero")
    B_o1 = tl.load(B_o1_ptr, boundary_check=(0, 1), padding_option="zero")
    B_o2 = tl.load(B_o2_ptr, boundary_check=(0, 1), padding_option="zero")
    B_d0 = tl.load(B_d0_ptr, boundary_check=(0, 1), padding_option="zero")
    B_d1 = tl.load(B_d1_ptr, boundary_check=(0, 1), padding_option="zero")
    B_d2 = tl.load(B_d2_ptr, boundary_check=(0, 1), padding_option="zero")
    B_ra = tl.load(i_B_ra_ptr, boundary_check=(0, 1), padding_option="zero")

    i_B_do0 = tl.load(i_B_do0_ptr, boundary_check=(0, 1), padding_option="zero")
    i_B_do1 = tl.load(i_B_do1_ptr, boundary_check=(0, 1), padding_option="zero")
    i_B_do2 = tl.load(i_B_do2_ptr, boundary_check=(0, 1), padding_option="zero")
    i_B_dra = tl.load(i_B_dra_ptr, boundary_check=(0, 1), padding_option="zero")

    B_prev_t_middle = tl.load(i_B_prev_t_ptr, boundary_check=(0, 1))

    d, k, c = tl.load(d_ptr), tl.load(k_ptr), tl.load(c_ptr)
    ai2, ai4, ai6, ai8, ai10, ai12 = tl.load(ai2_ptr), tl.load(ai4_ptr), tl.load(ai6_ptr), tl.load(ai8_ptr), tl.load(ai10_ptr), tl.load(ai12_ptr)

    t = B_prev_t_middle
    new_o0 = B_o0 + t * B_d0
    new_o1 = B_o1 + t * B_d1
    new_o2 = B_o2 + t * B_d2
    r2 = new_o0 * new_o0 + new_o1 * new_o1
    # sag
    if k > -1:
        valid = r2 < (1 / (c * c) / (1 + k))
    else:
        valid = (tl.zeros_like(B_o0) + 1).to(tl.int1)
    valid = valid & (B_ra > 0)
    # sag
    valid_x, valid_y = new_o0 * valid, new_o1 * valid
    valid_r2 = valid_x * valid_x + valid_y * valid_y
    valid_r2_2 = valid_r2 * valid_r2
    valid_r2_3, valid_r2_4 = valid_r2_2 * valid_r2, valid_r2_2 * valid_r2_2
    valid_r2_5, valid_r2_6 = valid_r2_3 * valid_r2_2, valid_r2_3 * valid_r2_3
    sf = tl.sqrt(1 - (1 + k) * valid_r2 * c * c + EPSILON)
    ts = valid_r2 * c / (1 + sf)
    ts_1 = ts + ai2 * valid_r2 + ai4 * valid_r2_2 + ai6 * valid_r2_3 + ai8 * valid_r2_4 + ai10 * valid_r2_5 + ai12 * valid_r2_6
    ft = ts_1 + d - new_o2

    dsdr2 = (1 + sf + (1 + k) * valid_r2 * c * c / 2 / sf) * c / (1 + sf) / (1 + sf)
    dsdr2_1 = dsdr2 + ai2 + 2 * ai4 * valid_r2 + 3 * ai6 * valid_r2_2 + 4 * ai8 * valid_r2_3 + 5 * ai10 * valid_r2_4 + 6 * ai12 * valid_r2_5
    dfdx, dfdy = 2 * dsdr2_1 * valid_x, 2 * dsdr2_1 * valid_y
    dfdz = tl.zeros_like(valid_x) - 1
    dfdt = dfdx * B_d0 + dfdy * B_d1 + dfdz * B_d2
    ft_dfdt = ft / (dfdt + EPSILON)
    final_t = t - tl.clamp(ft_dfdt, -step_bound, step_bound)

    fvalid = valid & (valid_r2 <= r * r)
    fvalid_x, fvalid_y = new_o0 * fvalid, new_o1 * fvalid
    fvalid_r2 = fvalid_x * fvalid_x + fvalid_y * fvalid_y
    fsf = tl.sqrt(1 - (1 + k) * fvalid_r2 * c * c + EPSILON)
    fts = fvalid_r2 * c / (1 + fsf)
    fts += (
        ai2 * fvalid_r2
        + ai4 * fvalid_r2 * fvalid_r2
        + ai6 * fvalid_r2 * fvalid_r2 * fvalid_r2
        + ai8 * fvalid_r2 * fvalid_r2 * fvalid_r2 * fvalid_r2
        + ai10 * fvalid_r2 * fvalid_r2 * fvalid_r2 * fvalid_r2 * fvalid_r2
        + ai12 * fvalid_r2 * fvalid_r2 * fvalid_r2 * fvalid_r2 * fvalid_r2 * fvalid_r2
    )
    fft = fts + d - new_o2
    ffvalid = fvalid & (tl.abs(fft) < tol_t) & (final_t > 0)

    dnew_o0_d, dnew_o1_d, dnew_o2_d = B_d0 / B_d2, B_d1 / B_d2, 1
    dnew_o0_o2, dnew_o1_o2, dnew_o2_o2 = -B_d0 / B_d2, -B_d1 / B_d2, 0
    dnew_o0_d0, dnew_o1_d1 = t, t
    dnew_o0_d2, dnew_o1_d2, dnew_o2_d2 = -B_d0 * (d - B_o2) / B_d2 / B_d2, -B_d1 * (d - B_o2) / B_d2 / B_d2, t - (d - B_o2) / B_d2
    dvalidx_d, dvalidy_d = dnew_o0_d * valid, dnew_o1_d * valid
    dvalidx_o2, dvalidy_o2 = valid * dnew_o0_o2, valid * dnew_o1_o2
    dvalidx_o0, dvalidy_o1 = valid, valid
    dvalidx_d0, dvalidy_d1 = valid * dnew_o0_d0, valid * dnew_o1_d1
    dvalidx_d2, dvalidy_d2 = valid * dnew_o0_d2, valid * dnew_o1_d2
    pvalid_r2_validx, pvalid_r2_validy = 2 * valid_x, 2 * valid_y
    dvalid_r2_o0, dvalid_r2_o1 = pvalid_r2_validx * dvalidx_o0, pvalid_r2_validy * dvalidy_o1
    dvalid_r2_o2 = pvalid_r2_validx * dvalidx_o2 + pvalid_r2_validy * dvalidy_o2
    dvalid_r2_d0, dvalid_r2_d1 = pvalid_r2_validx * dvalidx_d0, pvalid_r2_validy * dvalidy_d1
    dvalid_r2_d2 = pvalid_r2_validx * dvalidx_d2 + pvalid_r2_validy * dvalidy_d2
    dvalid_r2_d = pvalid_r2_validx * dvalidx_d + pvalid_r2_validy * dvalidy_d

    psf_valid_r2 = -1 / 2 / sf * (1 + k) * c * c
    dsf_d = psf_valid_r2 * dvalid_r2_d
    dsf_c = -1 / sf * (1 + k) * valid_r2 * c
    dsf_k = -1 / 2 / sf * valid_r2 * c * c
    dsf_o0, dsf_o1, dsf_o2 = psf_valid_r2 * dvalid_r2_o0, psf_valid_r2 * dvalid_r2_o1, psf_valid_r2 * dvalid_r2_o2
    dsf_d0, dsf_d1, dsf_d2 = psf_valid_r2 * dvalid_r2_d0, psf_valid_r2 * dvalid_r2_d1, psf_valid_r2 * dvalid_r2_d2
    pts_valid_r2 = c / (1 + sf)
    pts_sf = -1 / (1 + sf) / (1 + sf) * valid_r2 * c
    pts_c = valid_r2 / (1 + sf)
    dts_d = pts_valid_r2 * dvalid_r2_d + pts_sf * dsf_d
    dts_o0, dts_o1, dts_o2 = (
        pts_valid_r2 * dvalid_r2_o0 + pts_sf * dsf_o0,
        pts_valid_r2 * dvalid_r2_o1 + pts_sf * dsf_o1,
        pts_valid_r2 * dvalid_r2_o2 + pts_sf * dsf_o2,
    )
    dts_d0, dts_d1, dts_d2 = (
        pts_valid_r2 * dvalid_r2_d0 + pts_sf * dsf_d0,
        pts_valid_r2 * dvalid_r2_d1 + pts_sf * dsf_d1,
        pts_valid_r2 * dvalid_r2_d2 + pts_sf * dsf_d2,
    )
    dts_c = pts_c + pts_sf * dsf_c

    pts_1_ts = 1
    pts_1_valid_r2 = ai2 + 2 * ai4 * valid_r2 + 3 * ai6 * valid_r2_2 + 4 * ai8 * valid_r2_4 + 5 * ai10 * valid_r2_4 + 6 * ai12 * valid_r2_5
    dts_1_c = pts_1_ts * dts_c
    dts_1_d = pts_1_ts * dts_d + pts_1_valid_r2 * dvalid_r2_d
    dts_1_o0, dts_1_o1, dts_1_o2 = (
        pts_1_ts * dts_o0 + pts_1_valid_r2 * dvalid_r2_o0,
        pts_1_ts * dts_o1 + pts_1_valid_r2 * dvalid_r2_o1,
        pts_1_ts * dts_o2 + pts_1_valid_r2 * dvalid_r2_o2,
    )
    dts_1_d0, dts_1_d1, dts_1_d2 = (
        pts_1_ts * dts_d0 + pts_1_valid_r2 * dvalid_r2_d0,
        pts_1_ts * dts_d1 + pts_1_valid_r2 * dvalid_r2_d1,
        pts_1_ts * dts_d2 + pts_1_valid_r2 * dvalid_r2_d1,
    )
    pft_ts_1, pft_d, pft_new_o2 = 1, 1, -1
    dft_c = pft_ts_1 * dts_1_c
    dft_d = pft_ts_1 * dts_1_d + pft_d * 1 + pft_new_o2 * dnew_o2_d
    dft_o0, dft_o1, dft_o2 = (
        pft_ts_1 * dts_1_o0,
        pft_ts_1 * dts_1_o1,
        pft_ts_1 * dts_1_o2 + pft_new_o2 * dnew_o2_o2,
    )
    dft_d0, dft_d1, dft_d2 = (
        pft_ts_1 * dts_1_d0,
        pft_ts_1 * dts_1_d1,
        pft_ts_1 * dts_1_d2 + pft_new_o2 * dnew_o2_d2,
    )
    pdsdr2_sf = (1 - (1 + k) / 2 * valid_r2 * c * c / sf / sf) * c / (1 + sf) / (1 + sf) - 2 * c / (1 + sf) / (1 + sf) / (1 + sf) * (
        1 + sf + (1 + k) * valid_r2 * c * c / 2 / sf
    )
    pdsdr2_valid_r2 = c / (1 + sf) / (1 + sf) * ((1 + k) * c * c / 2 / sf)

    pdsdr2_c = 1 / (1 + sf) / (1 + sf) * (1 + sf + (1 + k) * valid_r2 * c * c / 2 / sf) + c / (1 + sf) / (1 + sf) * ((1 + k) * valid_r2 * c / sf)
    ddsdr2_c = pdsdr2_c + pdsdr2_sf * dsf_c
    ddsdr2_d = pdsdr2_sf * dsf_d + pdsdr2_valid_r2 * dvalid_r2_d
    ddsdr2_o0, ddsdr2_o1, ddsdr2_o2 = (
        pdsdr2_sf * dsf_o0 + pdsdr2_valid_r2 * dvalid_r2_o0,
        pdsdr2_sf * dsf_o1 + pdsdr2_valid_r2 * dvalid_r2_o1,
        pdsdr2_sf * dsf_o2 + pdsdr2_valid_r2 * dvalid_r2_o2,
    )
    ddsdr2_d0, ddsdr2_d1, ddsdr2_d2 = (
        pdsdr2_sf * dsf_d0 + pdsdr2_valid_r2 * dvalid_r2_d0,
        pdsdr2_sf * dsf_d1 + pdsdr2_valid_r2 * dvalid_r2_d1,
        pdsdr2_sf * dsf_d2 + pdsdr2_valid_r2 * dvalid_r2_d2,
    )
    pdsdr2_1_valid_r2 = 2 * ai4 + 6 * ai6 * valid_r2 + 12 * ai8 * valid_r2_2 + 20 * ai10 * valid_r2_3 + 30 * ai12 * valid_r2_4
    pdsdr2_1_dsdr2 = 1

    ddsdr2_1_d = pdsdr2_1_valid_r2 * dvalid_r2_d + pdsdr2_1_dsdr2 * ddsdr2_d

    ddsdr2_1_c = pdsdr2_1_dsdr2 * ddsdr2_c
    ddsdr2_1_o0, ddsdr2_1_o1, ddsdr2_1_o2 = (
        pdsdr2_1_dsdr2 * ddsdr2_o0 + pdsdr2_1_valid_r2 * dvalid_r2_o0,
        pdsdr2_1_dsdr2 * ddsdr2_o1 + pdsdr2_1_valid_r2 * dvalid_r2_o1,
        pdsdr2_1_dsdr2 * ddsdr2_o2 + pdsdr2_1_valid_r2 * dvalid_r2_o2,
    )
    ddsdr2_1_d0, ddsdr2_1_d1, ddsdr2_1_d2 = (
        pdsdr2_1_dsdr2 * ddsdr2_d0 + pdsdr2_1_valid_r2 * dvalid_r2_d0,
        pdsdr2_1_dsdr2 * ddsdr2_d1 + pdsdr2_1_valid_r2 * dvalid_r2_d1,
        pdsdr2_1_dsdr2 * ddsdr2_d2 + pdsdr2_1_valid_r2 * dvalid_r2_d2,
    )
    pdfdx_dsdr2_1, pdfdx_validx, pdfdy_dsdr2_1, pdfdy_validy = 2 * valid_x, 2 * dsdr2_1, 2 * valid_y, 2 * dsdr2_1
    ddfdx_c, ddfdy_c = pdfdx_dsdr2_1 * ddsdr2_1_c, 2 * valid_y * ddsdr2_1_c
    ddfdx_d, ddfdy_d = pdfdx_dsdr2_1 * ddsdr2_1_d + pdfdx_validx * dvalidx_d, pdfdy_dsdr2_1 * ddsdr2_1_d + pdfdy_validy * dvalidy_d
    ddfdx_o0, ddfdy_o0 = pdfdx_dsdr2_1 * ddsdr2_1_o0 + pdfdx_validx * dvalidx_o0, pdfdy_dsdr2_1 * ddsdr2_1_o0
    ddfdx_o1, ddfdy_o1 = pdfdx_dsdr2_1 * ddsdr2_1_o1, pdfdy_dsdr2_1 * ddsdr2_1_o1 + pdfdy_validy * dvalidy_o1
    ddfdx_o2, ddfdy_o2 = pdfdx_dsdr2_1 * ddsdr2_1_o2 + pdfdx_validx * dvalidx_o2, pdfdy_dsdr2_1 * ddsdr2_1_o2 + pdfdy_validy * dvalidy_o2
    ddfdx_d0, ddfdy_d0 = pdfdx_dsdr2_1 * ddsdr2_1_d0 + pdfdx_validx * dvalidx_d0, pdfdy_dsdr2_1 * ddsdr2_1_d0
    ddfdx_d1, ddfdy_d1 = pdfdx_dsdr2_1 * ddsdr2_1_d1, pdfdy_dsdr2_1 * ddsdr2_1_d1 + pdfdy_validy * dvalidy_d1
    ddfdx_d2, ddfdy_d2 = pdfdx_dsdr2_1 * ddsdr2_1_d2 + pdfdx_validx * dvalidx_d2, pdfdy_dsdr2_1 * ddsdr2_1_d2 + pdfdy_validy * dvalidy_d2

    ddfdt_c = B_d0 * ddfdx_c + B_d1 * ddfdy_c
    ddfdt_d = B_d0 * ddfdx_d + B_d1 * ddfdy_d
    ddfdt_o0, ddfdt_o1, ddfdt_o2 = B_d0 * ddfdx_o0 + B_d1 * ddfdy_o0, B_d0 * ddfdx_o1 + B_d1 * ddfdy_o1, B_d0 * ddfdx_o2 + B_d1 * ddfdy_o2
    ddfdt_d0, ddfdt_d1, ddfdt_d2 = (
        B_d0 * ddfdx_d0 + dfdx + B_d1 * ddfdy_d0,
        B_d0 * ddfdx_d1 + B_d1 * ddfdy_d1 + dfdy,
        B_d0 * ddfdx_d2 + B_d1 * ddfdy_d2 + dfdz,
    )

    # ddfdx_validx, ddfdy_validy = 2 * dsdr2_1, 2 * dsdr2_1

    dclamp_ft_dfdt = tl.where(tl.abs(ft_dfdt) < step_bound, 1, 0)

    pclamp_ft = dclamp_ft_dfdt / (dfdt + EPSILON)
    pclamp_dfdt = -1 * dclamp_ft_dfdt * ft / (dfdt + EPSILON) / (dfdt + EPSILON)

    pfinalt_clamp = -1
    # dt_d = 1 / d2 - (dt_ft * dft_d + dt_dfdt * dfdt_d)
    dclamp_d = pclamp_ft * dft_d + pclamp_dfdt * ddfdt_d
    dclamp_c = pclamp_ft * dft_c + pclamp_dfdt * ddfdt_c
    dclamp_o0 = pclamp_ft * dft_o0 + pclamp_dfdt * ddfdt_o0
    dclamp_o1 = pclamp_ft * dft_o1 + pclamp_dfdt * ddfdt_o1
    dclamp_o2 = pclamp_ft * dft_o2 + pclamp_dfdt * ddfdt_o2
    dclamp_d0 = pclamp_ft * dft_d0 + pclamp_dfdt * ddfdt_d0
    dclamp_d1 = pclamp_ft * dft_d1 + pclamp_dfdt * ddfdt_d1
    dclamp_d2 = pclamp_ft * dft_d2 + pclamp_dfdt * ddfdt_d2

    dfinal_t_d = 1 / (B_d2) + pfinalt_clamp * dclamp_d

    # tl.device_print("d", dt_d)
    dfinal_t_c = pfinalt_clamp * dclamp_c
    dfinal_t_o0, dfinal_t_o1, dfinal_t_o2 = pfinalt_clamp * dclamp_o0, pfinalt_clamp * dclamp_o1, (-1 / B_d2 + pfinalt_clamp * dclamp_o2)
    dfinal_t_d0, dfinal_t_d1, dfinal_t_d2 = (
        pfinalt_clamp * dclamp_d0,
        pfinalt_clamp * dclamp_d1,
        (-1 * (d - B_o2) / B_d2 / B_d2 + pfinalt_clamp * dclamp_d2),
    )

    pfnewo0_final_t, pfnewo0_B_o0, pfnewo0_B_d0 = B_d0, 1, final_t

    dfnewo0_d = pfnewo0_final_t * dfinal_t_d
    dfnewo0_c = pfnewo0_final_t * dfinal_t_c
    dfnewo0_o0 = pfnewo0_final_t * dfinal_t_o0 + pfnewo0_B_o0
    dfnewo0_o1 = pfnewo0_final_t * dfinal_t_o1
    dfnewo0_o2 = pfnewo0_final_t * dfinal_t_o2
    dfnewo0_d0 = pfnewo0_final_t * dfinal_t_d0 + pfnewo0_B_d0
    dfnewo0_d1 = pfnewo0_final_t * dfinal_t_d1
    dfnewo0_d2 = pfnewo0_final_t * dfinal_t_d2

    pfnewo1_final_t, pfnewo1_B_o1, pfnewo1_B_d1 = B_d1, 1, final_t
    dfnewo1_c = pfnewo1_final_t * dfinal_t_c
    dfnewo1_d = pfnewo1_final_t * dfinal_t_d
    dfnewo1_o0 = pfnewo1_final_t * dfinal_t_o0
    dfnewo1_o1 = pfnewo1_final_t * dfinal_t_o1 + pfnewo1_B_o1
    dfnewo1_o2 = pfnewo1_final_t * dfinal_t_o2
    dfnewo1_d0 = pfnewo1_final_t * dfinal_t_d0
    dfnewo1_d1 = pfnewo1_final_t * dfinal_t_d1 + pfnewo1_B_d1
    dfnewo1_d2 = pfnewo1_final_t * dfinal_t_d2

    pfnewo2_final_t, pfnewo2_B_o2, pfnewo2_B_d2 = B_d2, 1, final_t
    dfnewo2_c = pfnewo2_final_t * dfinal_t_c
    dfnewo2_d = pfnewo2_final_t * dfinal_t_d
    dfnewo2_o0 = pfnewo2_final_t * dfinal_t_o0
    dfnewo2_o1 = pfnewo2_final_t * dfinal_t_o1
    dfnewo2_o2 = pfnewo2_final_t * dfinal_t_o2 + pfnewo2_B_o2
    dfnewo2_d0 = pfnewo2_final_t * dfinal_t_d0
    dfnewo2_d1 = pfnewo2_final_t * dfinal_t_d1
    dfnewo2_d2 = pfnewo2_final_t * dfinal_t_d2 + pfnewo2_B_d2
    vfselect = tl.where(ffvalid, 1, 0)
    vfnselect = tl.where(ffvalid, 0, 1)

    dvfnewo0_d, dvfnewo0_c = vfselect * dfnewo0_d, vfselect * dfnewo0_c
    dvfnewo0_o0, dvfnewo0_o1, dvfnewo0_o2 = vfselect * dfnewo0_o0 + vfnselect, vfselect * dfnewo0_o1 + vfnselect, vfselect * dfnewo0_o2 + vfnselect
    dvfnewo0_d0, dvfnewo0_d1, dvfnewo0_d2 = vfselect * dfnewo0_d0, vfselect * dfnewo0_d1, vfselect * dfnewo0_d2

    dvfnewo1_d, dvfnewo1_c = vfselect * dfnewo1_d, vfselect * dfnewo1_c
    dvfnewo1_o0, dvfnewo1_o1, dvfnewo1_o2 = vfselect * dfnewo1_o0 + vfnselect, vfselect * dfnewo1_o1 + vfnselect, vfselect * dfnewo1_o2 + vfnselect
    dvfnewo1_d0, dvfnewo1_d1, dvfnewo1_d2 = vfselect * dfnewo1_d0, vfselect * dfnewo1_d1, vfselect * dfnewo1_d2

    dvfnewo2_d, dvfnewo2_c = vfselect * dfnewo2_d, vfselect * dfnewo2_c
    dvfnewo2_o0, dvfnewo2_o1, dvfnewo2_o2 = vfselect * dfnewo2_o0 + vfnselect, vfselect * dfnewo2_o1 + vfnselect, vfselect * dfnewo2_o2 + vfnselect
    dvfnewo2_d0, dvfnewo2_d1, dvfnewo2_d2 = vfselect * dfnewo2_d0, vfselect * dfnewo2_d1, vfselect * dfnewo2_d2

    d_c = i_B_do0 * dvfnewo0_c + i_B_do1 * dvfnewo1_c + i_B_do2 * dvfnewo2_c
    d_d = i_B_do0 * dvfnewo0_d + i_B_do1 * dvfnewo1_d + i_B_do2 * dvfnewo2_d
    d_o0 = i_B_do0 * dvfnewo0_o0 + i_B_do1 * dvfnewo1_o0 + i_B_do2 * dvfnewo2_o0
    d_o1 = i_B_do0 * dvfnewo0_o1 + i_B_do1 * dvfnewo1_o1 + i_B_do2 * dvfnewo2_o1
    d_o2 = i_B_do0 * dvfnewo0_o2 + i_B_do1 * dvfnewo1_o2 + i_B_do2 * dvfnewo2_o2
    d_d0 = i_B_do0 * dvfnewo0_d0 + i_B_do1 * dvfnewo1_d0 + i_B_do2 * dvfnewo2_d0
    d_d1 = i_B_do0 * dvfnewo0_d1 + i_B_do1 * dvfnewo1_d1 + i_B_do2 * dvfnewo2_d1
    d_d2 = i_B_do0 * dvfnewo0_d2 + i_B_do1 * dvfnewo1_d2 + i_B_do2 * dvfnewo2_d2
    ###ai
    ddfdx_dsdr2_1, ddfdy_dsdr2_1 = 2 * valid_x, 2 * valid_y
    dfdt_dsdr2_1 = B_d0 * ddfdx_dsdr2_1 + B_d1 * ddfdy_dsdr2_1
    pclamp_dsdr2_1 = pclamp_dfdt * dfdt_dsdr2_1
    dfnewo0_clamp, dfnewo1_clamp, dfnewo2_clamp = B_d0 * pfinalt_clamp, B_d1 * pfinalt_clamp, B_d2 * pfinalt_clamp
    dvfnewo0_clamp, dvfnewo1_clamp, dvfnewo2_clamp = vfselect * dfnewo0_clamp, vfselect * dfnewo1_clamp, vfselect * dfnewo2_clamp
    d_clamp = i_B_do0 * dvfnewo0_clamp + i_B_do1 * dvfnewo1_clamp + i_B_do2 * dvfnewo2_clamp


    d_ai2 = d_clamp * (pclamp_ft * valid_r2 + pclamp_dsdr2_1)
    d_ai4 = d_clamp * (pclamp_ft * valid_r2_2 + pclamp_dsdr2_1 * 2 * valid_r2)
    d_ai6 = d_clamp * (pclamp_ft * valid_r2_3 + pclamp_dsdr2_1 * 3 * valid_r2_2)
    d_ai8 = d_clamp * (pclamp_ft * valid_r2_4 + pclamp_dsdr2_1 * 4 * valid_r2_3)
    d_ai10 = d_clamp * (pclamp_ft * valid_r2_5 + pclamp_dsdr2_1 * 5 * valid_r2_4)
    d_ai12 = d_clamp * (pclamp_ft * valid_r2_6 + pclamp_dsdr2_1 * 6 * valid_r2_5)

    d_ra = ffvalid * i_B_dra
    ####

    tl.store(O_B_do0_ptr, d_o0, boundary_check=(0, 1))
    tl.store(O_B_do1_ptr, d_o1, boundary_check=(0, 1))
    tl.store(O_B_do2_ptr, d_o2, boundary_check=(0, 1))
    tl.store(O_B_dd0_ptr, d_d0, boundary_check=(0, 1))
    tl.store(O_B_dd1_ptr, d_d1, boundary_check=(0, 1))
    tl.store(O_B_dd2_ptr, d_d2, boundary_check=(0, 1))
    tl.store(O_B_dra_ptr, d_ra, boundary_check=(0, 1))

    dt_d = tl.where(mask, d_d, 0)
    dt_c = tl.where(mask, d_c, 0)
    dt_dai2 = tl.where(mask, d_ai2, 0)
    dt_dai4 = tl.where(mask, d_ai4, 0)
    dt_dai6 = tl.where(mask, d_ai6, 0)
    dt_dai8 = tl.where(mask, d_ai8, 0)
    dt_dai10 = tl.where(mask, d_ai10, 0)
    dt_dai12 = tl.where(mask, d_ai12, 0)
    # tl.device_print("mask d", dt_d)
    while tl.atomic_cas(LOCK, 0, 1) == 1:
        pass

    dt_dai2_sum = tl.sum(dt_dai2)
    dt_dai4_sum = tl.sum(dt_dai4)
    dt_dai6_sum = tl.sum(dt_dai6)
    dt_dai8_sum = tl.sum(dt_dai8)
    dt_dai10_sum = tl.sum(dt_dai10)
    dt_dai12_sum = tl.sum(dt_dai12)
    dt_d_sum = tl.sum(dt_d)
    dt_c_sum = tl.sum(dt_c)
    tl.store(o_ai2_ptr + pid, dt_dai2_sum)
    tl.store(o_ai4_ptr + pid, dt_dai4_sum)
    tl.store(o_ai6_ptr + pid, dt_dai6_sum)
    tl.store(o_ai8_ptr + pid, dt_dai8_sum)
    tl.store(o_ai10_ptr + pid, dt_dai10_sum)
    tl.store(o_ai12_ptr + pid, dt_dai12_sum)
    tl.store(o_d_ptr + pid, dt_d_sum)
    tl.store(o_c_ptr + pid, dt_c_sum)
    tl.atomic_xchg(LOCK, 0)


def flash_newtons_method_bwd(i_o, i_d, i_ra, i_prev_t, d_o, d_ra, d, k, c, r, ai2, ai4, ai6, ai8, ai10, ai12, tol_t, step_bound):

    # B, G1, G2, D = i_o.shape
    input_shape = i_o.shape
    i_o, i_d, i_ra = i_o.reshape(-1, input_shape[-1]), i_d.reshape(-1, input_shape[-1]), i_ra.reshape(-1, 1)
    T, D = i_o.shape
    o_do, o_dd, o_dra = torch.zeros_like(i_o), torch.zeros_like(i_d), torch.zeros_like(i_ra)

    BLOCK_SIZE = 512
    BN = triton.cdiv(T, BLOCK_SIZE)
    locks = torch.zeros((BN,), dtype=torch.int32, device=d.device)
    o_d = torch.zeros((BN,), device=d.device, dtype=d.dtype)
    o_k, o_c = torch.zeros_like(o_d), torch.zeros_like(o_d)
    o_ai2, o_ai4, o_ai6, o_ai8, o_ai10, o_ai12 = (
        torch.zeros_like(o_d),
        torch.zeros_like(o_d),
        torch.zeros_like(o_d),
        torch.zeros_like(o_d),
        torch.zeros_like(o_d),
        torch.zeros_like(o_d),
    )
    grid = (BN,)

    flash_newtons_method_bwd_kernel[grid](
        i_o,
        i_d,
        i_ra,
        i_prev_t,
        d_o,
        d_ra,
        # d_t,
        # d_valid,
        d,
        k,
        c,
        r,
        ai2,
        ai4,
        ai6,
        ai8,
        ai10,
        ai12,
        locks,
        o_do,
        o_dd,
        o_dra,
        o_d,
        o_k,
        o_c,
        o_ai2,
        o_ai4,
        o_ai6,
        o_ai8,
        o_ai10,
        o_ai12,
        tol_t,
        step_bound,
        T,
        D,
        BS=BLOCK_SIZE,
    )

    o_do, o_dd, o_dra = o_do.reshape(input_shape), o_dd.reshape(input_shape), o_dra.reshape(input_shape[:-1])
    return o_do, o_dd, o_dra, o_d, o_k, o_c, o_ai2, o_ai4, o_ai6, o_ai8, o_ai10, o_ai12


class FlashNewtonsMethodFunction(torch.autograd.Function):
    @staticmethod
    @contiguous
    @autocast_custom_fwd
    def forward(ctx, i_o, i_d, i_ra, d, k, c, r, ai2, ai4, ai6, ai8, ai10, ai12, tol_i, tol_t, max_iter, step_bound):

        new_o, new_ra, prev_t = flash_newtons_method_fwd(
            i_o, i_d, i_ra, d, k, c, r, ai2, ai4, ai6, ai8, ai10, ai12, tol_i, tol_t, max_iter, step_bound
        )

        ctx.save_for_backward(i_o, i_d, i_ra, prev_t, d, k, c, ai2, ai4, ai6, ai8, ai10, ai12)
        ctx.r, ctx.tol_i, ctx.tol_t, ctx.step_bound = r, tol_i, tol_t, step_bound
        return new_o, new_ra

    @staticmethod
    @contiguous
    @autocast_custom_bwd
    def backward(ctx, d_o, d_ra):
        i_o, i_d, i_ra, prev_t, d, k, c, ai2, ai4, ai6, ai8, ai10, ai12 = ctx.saved_tensors
        r, tol_t, step_bound = ctx.r, ctx.tol_t, ctx.step_bound
        o_do, o_dd, o_dra, o_d, o_k, o_c, o_ai2, o_ai4, o_ai6, o_ai8, o_ai10, o_ai12 = flash_newtons_method_bwd(
            i_o, i_d, i_ra, prev_t, d_o, d_ra, d, k, c, r, ai2, ai4, ai6, ai8, ai10, ai12, tol_t, step_bound
        )

        dt_ai2, dt_ai4, dt_ai6, dt_ai8, dt_ai10, dt_ai12, dt_d, dt_c = (
            torch.sum(o_ai2, 0, keepdim=True),
            torch.sum(o_ai4, 0, keepdim=True),
            torch.sum(o_ai6, 0, keepdim=True),
            torch.sum(o_ai8, 0, keepdim=True),
            torch.sum(o_ai10, 0, keepdim=True),
            torch.sum(o_ai12, 0, keepdim=True),
            torch.sum(o_d, 0, keepdim=True),
            torch.sum(o_c, 0, keepdim=True),
        )
        return o_do, o_dd, o_dra, dt_d, None, dt_c, None, dt_ai2, dt_ai4, dt_ai6, dt_ai8, dt_ai10, dt_ai12, None, None, None, None


@torch.compiler.disable
def flash_newtons_method(i_o, i_d, i_ra, d, k, c, r, ai2, ai4, ai6, ai8, ai10, ai12, tol_i, tol_t, max_iter, step_bound):
    new_o, new_ra = FlashNewtonsMethodFunction.apply(i_o, i_d, i_ra, d, k, c, r, ai2, ai4, ai6, ai8, ai10, ai12, tol_i, tol_t, max_iter, step_bound)
    return new_o, new_ra
