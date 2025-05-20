import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as nnF
import triton
import triton.language as tl

from .utils import *

BKV_LIST = [
    512,
    1024,
    2048,
    # 16,
]

EPSILON: tl.constexpr = epsilon


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
def flash_refaction_fwd_kernel(
    i_o_ptr,  # (T,3)
    i_d_ptr,  # (T,3)
    i_ra_ptr,  # (T,1)
    i_obliq_ptr,  # (T,1)
    o_d_ptr,  # (T,3)
    o_ra_ptr,
    o_obliq_ptr,  # (T,1)
    o_valid_ptr,
    i_eta_ptr,  # (C)
    d_ptr,  # (1)
    k_ptr,  # (1)
    c_ptr,  # (1)
    ai2_ptr,  # (1)
    ai4_ptr,  # (1)
    ai6_ptr,  # (1)
    ai8_ptr,  # (1)
    ai10_ptr,  # (1)
    ai12_ptr,  # (1)
    C,
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
    B_obliq_ptr = tl.make_block_ptr(i_obliq_ptr, (T, 1), (1, 1), (pid * BS, 0), (BS, 1), order=(1, 0))

    O_B_d0_ptr = tl.make_block_ptr(o_d_ptr, (T, D), (D, 1), (pid * BS, 0), (BS, 1), order=(1, 0))
    O_B_d1_ptr = tl.make_block_ptr(o_d_ptr, (T, D), (D, 1), (pid * BS, 1), (BS, 1), order=(1, 0))
    O_B_d2_ptr = tl.make_block_ptr(o_d_ptr, (T, D), (D, 1), (pid * BS, 2), (BS, 1), order=(1, 0))

    O_B_ra_ptr = tl.make_block_ptr(o_ra_ptr, (T, 1), (1, 1), (pid * BS, 0), (BS, 1), order=(1, 0))
    O_B_obliq_ptr = tl.make_block_ptr(o_obliq_ptr, (T, 1), (1, 1), (pid * BS, 0), (BS, 1), order=(1, 0))
    O_B_valid_ptr = tl.make_block_ptr(o_valid_ptr, (T, 1), (1, 1), (pid * BS, 0), (BS, 1), order=(1, 0))

    B_o0 = tl.load(B_o0_ptr, boundary_check=(0, 1), padding_option="zero")
    B_o1 = tl.load(B_o1_ptr, boundary_check=(0, 1), padding_option="zero")
    B_o2 = tl.load(B_o2_ptr, boundary_check=(0, 1), padding_option="zero")
    B_d0 = tl.load(B_d0_ptr, boundary_check=(0, 1), padding_option="zero")
    B_d1 = tl.load(B_d1_ptr, boundary_check=(0, 1), padding_option="zero")
    B_d2 = tl.load(B_d2_ptr, boundary_check=(0, 1), padding_option="zero")
    B_ra = tl.load(B_ra_ptr, boundary_check=(0, 1), padding_option="zero")
    B_obliq = tl.load(B_obliq_ptr, boundary_check=(0, 1), padding_option="zero")

    d, k, c = tl.load(d_ptr), tl.load(k_ptr), tl.load(c_ptr)
    ai2, ai4, ai6, ai8, ai10, ai12 = tl.load(ai2_ptr), tl.load(ai4_ptr), tl.load(ai6_ptr), tl.load(ai8_ptr), tl.load(ai10_ptr), tl.load(ai12_ptr)
    offsets = pid * BS + tl.arange(0, BS)
    mask = offsets < T
    eta_offsets = offsets // (T // C)
    eta = tl.load(i_eta_ptr + eta_offsets, mask=mask)
    eta = tl.reshape(eta, (BS, 1))
    # dfdxyz
    r2 = B_o0 * B_o0 + B_o1 * B_o1
    if k > -1:
        valid = r2 < (1 / (c * c) / (1 + k))
    else:
        valid = (tl.zeros_like(B_o0) + 1).to(tl.int1)

    valid_x, valid_y = B_o0 * valid, B_o1 * valid
    valid_r2 = valid_x * valid_x + valid_y * valid_y
    sf = tl.sqrt(1 - (1 + k) * valid_r2 * c * c + EPSILON)
    dsdr2 = (1 + sf + (1 + k) * valid_r2 * c * c / 2 / sf) * c / (1 + sf) / (1 + sf)
    dsdr2_1 = (
        dsdr2
        + ai2
        + 2 * ai4 * valid_r2
        + 3 * ai6 * valid_r2 * valid_r2
        + 4 * ai8 * valid_r2 * valid_r2 * valid_r2
        + 5 * ai10 * valid_r2 * valid_r2 * valid_r2 * valid_r2
        + 6 * ai12 * valid_r2 * valid_r2 * valid_r2 * valid_r2 * valid_r2
    )
    dfdx, dfdy = 2 * dsdr2_1 * valid_x, 2 * dsdr2_1 * valid_y
    dfdz = tl.zeros_like(valid_x) - 1
    L2_norm_denom = tl.sqrt(dfdx * dfdx + dfdy * dfdy + dfdz * dfdz)
    norm_x, norm_y, norm_z = dfdx / L2_norm_denom, dfdy / L2_norm_denom, dfdz / L2_norm_denom
    forward = tl.sum(B_d2 * B_ra) > 0
    if forward:
        norm_x, norm_y, norm_z = -norm_x, -norm_y, -norm_z
    cosi = B_d0 * norm_x + B_d1 * norm_y + B_d2 * norm_z
    valid_tir = (eta * eta * (1 - cosi * cosi) < 1) & (B_ra > 0)

    sr = tl.sqrt(1 - eta * eta * (1 - cosi * cosi) * valid_tir + EPSILON)

    d0 = sr * norm_x + eta * (B_d0 - cosi * norm_x)
    d1 = sr * norm_y + eta * (B_d1 - cosi * norm_y)
    d2 = sr * norm_z + eta * (B_d2 - cosi * norm_z)
    new_d0 = tl.where(valid_tir, d0, B_d0)
    new_d1 = tl.where(valid_tir, d1, B_d1)
    new_d2 = tl.where(valid_tir, d2, B_d2)

    new_ra = B_ra * valid_tir
    obliq = new_d0 * B_d0 + new_d1 * B_d1 + new_d2 * B_d2
    new_obliq = tl.where(valid_tir, obliq, B_obliq)

    tl.store(O_B_d0_ptr, new_d0, boundary_check=(0, 1))
    tl.store(O_B_d1_ptr, new_d1, boundary_check=(0, 1))
    tl.store(O_B_d2_ptr, new_d2, boundary_check=(0, 1))

    tl.store(O_B_ra_ptr, new_ra, boundary_check=(0, 1))
    tl.store(O_B_obliq_ptr, new_obliq, boundary_check=(0, 1))
    tl.store(O_B_valid_ptr, valid_tir.to(tl.int8), boundary_check=(0, 1))


def flash_refaction_fwd(i_o, i_d, i_ra, i_obliq, eta, d, k, c, ai2, ai4, ai6, ai8, ai10, ai12):

    input_shape = i_o.shape
    i_o, i_d, i_ra, i_obliq = i_o.reshape(-1, input_shape[-1]), i_d.reshape(-1, input_shape[-1]), i_ra.reshape(-1, 1), i_obliq.reshape(-1, 1)

    (T, D), (C,) = i_o.shape, eta.shape
    o_d, o_ra, o_obliq = torch.zeros_like(i_d), torch.zeros_like(i_ra), torch.zeros_like(i_obliq)
    o_valid = torch.zeros_like(i_ra, dtype=torch.bool)
    grid = lambda meta: (triton.cdiv(T, meta["BS"]),)
    flash_refaction_fwd_kernel[grid](i_o, i_d, i_ra, i_obliq, o_d, o_ra, o_obliq, o_valid, eta, d, k, c, ai2, ai4, ai6, ai8, ai10, ai12, C, T, D)
    o_d, o_ra, o_obliq, o_valid = (
        o_d.reshape(input_shape),
        o_ra.reshape(input_shape[:-1]),
        o_obliq.reshape(input_shape[:-1]),
        o_valid.reshape(input_shape[:-1]),
    )
    return o_d, o_ra, o_obliq, o_valid


@triton.jit
def flash_refaction_bwd_kernel(
    # fmt: off
    i_o_ptr, i_d_ptr, i_ra_ptr,i_obliq_ptr,i_dd_ptr,i_dra_ptr,i_dobliq_ptr,i_eta_ptr, d_ptr, k_ptr,c_ptr,
    ai2_ptr,ai4_ptr,ai6_ptr,ai8_ptr, ai10_ptr, ai12_ptr,locks_ptr,
    o_do_ptr, o_dd_ptr, o_dra_ptr, o_dobliq_ptr,o_d_ptr,o_k_ptr,o_c_ptr,o_ai2_ptr,o_ai4_ptr, o_ai6_ptr, o_ai8_ptr, o_ai10_ptr,o_ai12_ptr,
    C,T,D,BS: tl.constexpr
    # fmt: on
):

    pid = tl.program_id(0)
    LOCK = locks_ptr + pid
    B_o0_ptr = tl.make_block_ptr(i_o_ptr, (T, D), (D, 1), (pid * BS, 0), (BS, 1), order=(1, 0))
    B_o1_ptr = tl.make_block_ptr(i_o_ptr, (T, D), (D, 1), (pid * BS, 1), (BS, 1), order=(1, 0))
    B_o2_ptr = tl.make_block_ptr(i_o_ptr, (T, D), (D, 1), (pid * BS, 2), (BS, 1), order=(1, 0))
    B_d0_ptr = tl.make_block_ptr(i_d_ptr, (T, D), (D, 1), (pid * BS, 0), (BS, 1), order=(1, 0))
    B_d1_ptr = tl.make_block_ptr(i_d_ptr, (T, D), (D, 1), (pid * BS, 1), (BS, 1), order=(1, 0))
    B_d2_ptr = tl.make_block_ptr(i_d_ptr, (T, D), (D, 1), (pid * BS, 2), (BS, 1), order=(1, 0))
    B_dd0_ptr = tl.make_block_ptr(i_dd_ptr, (T, D), (D, 1), (pid * BS, 0), (BS, 1), order=(1, 0))
    B_dd1_ptr = tl.make_block_ptr(i_dd_ptr, (T, D), (D, 1), (pid * BS, 1), (BS, 1), order=(1, 0))
    B_dd2_ptr = tl.make_block_ptr(i_dd_ptr, (T, D), (D, 1), (pid * BS, 2), (BS, 1), order=(1, 0))

    B_ra_ptr = tl.make_block_ptr(i_ra_ptr, (T, 1), (1, 1), (pid * BS, 0), (BS, 1), order=(1, 0))
    B_obliq_ptr = tl.make_block_ptr(i_obliq_ptr, (T, 1), (1, 1), (pid * BS, 0), (BS, 1), order=(1, 0))
    B_dra_ptr = tl.make_block_ptr(i_dra_ptr, (T, 1), (1, 1), (pid * BS, 0), (BS, 1), order=(1, 0))
    B_dobliq_ptr = tl.make_block_ptr(i_dobliq_ptr, (T, 1), (1, 1), (pid * BS, 0), (BS, 1), order=(1, 0))

    O_B_do0_ptr = tl.make_block_ptr(o_do_ptr, (T, D), (D, 1), (pid * BS, 0), (BS, 1), order=(1, 0))
    O_B_do1_ptr = tl.make_block_ptr(o_do_ptr, (T, D), (D, 1), (pid * BS, 1), (BS, 1), order=(1, 0))
    O_B_do2_ptr = tl.make_block_ptr(o_do_ptr, (T, D), (D, 1), (pid * BS, 2), (BS, 1), order=(1, 0))
    O_B_dd0_ptr = tl.make_block_ptr(o_dd_ptr, (T, D), (D, 1), (pid * BS, 0), (BS, 1), order=(1, 0))
    O_B_dd1_ptr = tl.make_block_ptr(o_dd_ptr, (T, D), (D, 1), (pid * BS, 1), (BS, 1), order=(1, 0))
    O_B_dd2_ptr = tl.make_block_ptr(o_dd_ptr, (T, D), (D, 1), (pid * BS, 2), (BS, 1), order=(1, 0))
    O_B_dra_ptr = tl.make_block_ptr(o_dra_ptr, (T, 1), (1, 1), (pid * BS, 0), (BS, 1), order=(1, 0))
    O_B_dobliq_ptr = tl.make_block_ptr(o_dobliq_ptr, (T, 1), (1, 1), (pid * BS, 0), (BS, 1), order=(1, 0))

    mask = (pid * BS + tl.arange(0, BS))[:, None] < T
    B_o0 = tl.load(B_o0_ptr, boundary_check=(0, 1), padding_option="zero")
    B_o1 = tl.load(B_o1_ptr, boundary_check=(0, 1), padding_option="zero")
    B_o2 = tl.load(B_o2_ptr, boundary_check=(0, 1), padding_option="zero")
    B_d0 = tl.load(B_d0_ptr, boundary_check=(0, 1), padding_option="zero")
    B_d1 = tl.load(B_d1_ptr, boundary_check=(0, 1), padding_option="zero")
    B_d2 = tl.load(B_d2_ptr, boundary_check=(0, 1), padding_option="zero")
    B_ra = tl.load(B_ra_ptr, boundary_check=(0, 1), padding_option="zero")
    B_obliq = tl.load(B_obliq_ptr, boundary_check=(0, 1), padding_option="zero")
    B_dra = tl.load(B_dra_ptr, boundary_check=(0, 1), padding_option="zero")
    B_dobliq = tl.load(B_dobliq_ptr, boundary_check=(0, 1), padding_option="zero")

    B_dd0 = tl.load(B_dd0_ptr, boundary_check=(0, 1), padding_option="zero")
    B_dd1 = tl.load(B_dd1_ptr, boundary_check=(0, 1), padding_option="zero")
    B_dd2 = tl.load(B_dd2_ptr, boundary_check=(0, 1), padding_option="zero")

    # data_shape, block_shape = (S, G1, G2, D), (BS, G1, G2, 1)

    d, k, c = tl.load(d_ptr), tl.load(k_ptr), tl.load(c_ptr)
    ai2, ai4, ai6, ai8, ai10, ai12 = tl.load(ai2_ptr), tl.load(ai4_ptr), tl.load(ai6_ptr), tl.load(ai8_ptr), tl.load(ai10_ptr), tl.load(ai12_ptr)
    offsets = pid * BS + tl.arange(0, BS)
    eta_mask = offsets < T
    eta_offsets = offsets // (T // C)
    eta = tl.load(i_eta_ptr + eta_offsets, mask=eta_mask)
    eta = tl.reshape(eta, (BS, 1))
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
    dsdr2 = (1 + sf + (1 + k) * valid_r2 * c * c / 2 / sf) * c / (1 + sf) / (1 + sf)
    dsdr2_1 = dsdr2 + ai2 + 2 * ai4 * valid_r2 + 3 * ai6 * valid_r2_2 + 4 * ai8 * valid_r2_3 + 5 * ai10 * valid_r2_4 + 6 * ai12 * valid_r2_5

    dfdx, dfdy = 2 * dsdr2_1 * valid_x, 2 * dsdr2_1 * valid_y
    dfdz = tl.zeros_like(valid_x) - 1
    L2_norm_square = dfdx * dfdx + dfdy * dfdy + dfdz * dfdz
    L2_norm_denom = tl.sqrt(L2_norm_square)
    norm_x, norm_y, norm_z = dfdx / L2_norm_denom, dfdy / L2_norm_denom, dfdz / L2_norm_denom
    forward = tl.sum(B_d2 * B_ra) > 0
    if forward:
        norm_x, norm_y, norm_z = -norm_x, -norm_y, -norm_z
    cosi = B_d0 * norm_x + B_d1 * norm_y + B_d2 * norm_z
    valid_tir = (eta * eta * (1 - cosi * cosi) < 1) & (B_ra > 0)

    sr = tl.sqrt(1 - eta * eta * (1 - cosi * cosi) * valid_tir + EPSILON)

    d0 = sr * norm_x + eta * (B_d0 - cosi * norm_x)
    d1 = sr * norm_y + eta * (B_d1 - cosi * norm_y)
    d2 = sr * norm_z + eta * (B_d2 - cosi * norm_z)
    new_d0 = tl.where(valid_tir, d0, B_d0)
    new_d1 = tl.where(valid_tir, d1, B_d1)
    new_d2 = tl.where(valid_tir, d2, B_d2)

    new_ra = B_ra * valid_tir
    obliq = new_d0 * B_d0 + new_d1 * B_d1 + new_d2 * B_d2
    new_obliq = tl.where(valid_tir, obliq, B_obliq)

    dvalidx_o0, dvalidy_o1 = valid, valid
    dvalid_r2_o0, dvalid_r2_o1 = 2 * valid_x * dvalidx_o0, 2 * valid_y * dvalidy_o1
    psf_valid_r2 = -1 / 2 / sf * (1 + k) * c * c
    dsf_c = -1 / sf * (1 + k) * valid_r2 * c
    dsf_k = -1 / 2 / sf * valid_r2 * c * c
    dsf_o0, dsf_o1 = psf_valid_r2 * dvalid_r2_o0, psf_valid_r2 * dvalid_r2_o1
    pdsdr2_sf = (1 - (1 + k) / 2 * valid_r2 * c * c / sf / sf) * c / (1 + sf) / (1 + sf) - 2 * c / (1 + sf) / (1 + sf) / (1 + sf) * (
        1 + sf + (1 + k) * valid_r2 * c * c / 2 / sf
    )
    pdsdr2_valid_r2 = c / (1 + sf) / (1 + sf) * ((1 + k) * c * c / 2 / sf)
    pdsdr2_c = 1 / (1 + sf) / (1 + sf) * (1 + sf + (1 + k) * valid_r2 * c * c / 2 / sf) + c / (1 + sf) / (1 + sf) * ((1 + k) * valid_r2 * c / sf)
    ddsdr2_c = pdsdr2_c + pdsdr2_sf * dsf_c
    ddsdr2_o0, ddsdr2_o1 = (
        pdsdr2_sf * dsf_o0 + pdsdr2_valid_r2 * dvalid_r2_o0,
        pdsdr2_sf * dsf_o1 + pdsdr2_valid_r2 * dvalid_r2_o1,
    )
    pdsdr2_1_valid_r2 = 2 * ai4 + 6 * ai6 * valid_r2 + 12 * ai8 * valid_r2_2 + 20 * ai10 * valid_r2_3 + 30 * ai12 * valid_r2_4
    pdsdr2_1_dsdr2 = 1
    ddsdr2_1_c = pdsdr2_1_dsdr2 * ddsdr2_c
    ddsdr2_1_o0, ddsdr2_1_o1 = (
        pdsdr2_1_dsdr2 * ddsdr2_o0 + pdsdr2_1_valid_r2 * dvalid_r2_o0,
        pdsdr2_1_dsdr2 * ddsdr2_o1 + pdsdr2_1_valid_r2 * dvalid_r2_o1,
    )

    pdfdx_dsdr2_1, pdfdx_validx, pdfdy_dsdr2_1, pdfdy_validy = 2 * valid_x, 2 * dsdr2_1, 2 * valid_y, 2 * dsdr2_1
    ddfdx_c, ddfdy_c = pdfdx_dsdr2_1 * ddsdr2_1_c, 2 * valid_y * ddsdr2_1_c
    ddfdx_o0, ddfdy_o0 = pdfdx_dsdr2_1 * ddsdr2_1_o0 + pdfdx_validx * dvalidx_o0, pdfdy_dsdr2_1 * ddsdr2_1_o0
    ddfdx_o1, ddfdy_o1 = pdfdx_dsdr2_1 * ddsdr2_1_o1, pdfdy_dsdr2_1 * ddsdr2_1_o1 + pdfdy_validy * dvalidy_o1

    pdenom_dfdx, pdenom_dfdy, pdenom_dfdz = dfdx / L2_norm_denom, dfdy / L2_norm_denom, dfdz / L2_norm_denom
    pdenom_dsdr2_1 = pdenom_dfdx * pdfdx_dsdr2_1 + pdenom_dfdy * pdfdy_dsdr2_1
    ddenom_c = pdenom_dfdx * ddfdx_c + pdenom_dfdy * ddfdy_c
    ddenom_o0 = pdenom_dfdx * ddfdx_o0 + pdenom_dfdy * ddfdy_o0
    ddenom_o1 = pdenom_dfdx * ddfdx_o1 + pdenom_dfdy * ddfdy_o1

    pnormx_dfdx, pnormy_dfdy, pnormz_dfdz = 1 / L2_norm_denom, 1 / L2_norm_denom, 1 / L2_norm_denom
    pnormx_denom, pnormy_denom, pnormz_denom = -dfdx / L2_norm_square, -dfdy / L2_norm_square, -dfdz / L2_norm_square
    pnormx_dsdr2_1 = pnormx_denom * pdenom_dsdr2_1 + pnormx_dfdx * pdfdx_dsdr2_1
    pnormy_dsdr2_1 = pnormy_denom * pdenom_dsdr2_1 + pnormy_dfdy * pdfdy_dsdr2_1
    pnormz_dsdr2_1 = pnormz_denom * pdenom_dsdr2_1
    dnormx_c = pnormx_dfdx * ddfdx_c + pnormx_denom * ddenom_c
    dnormx_o0 = pnormx_dfdx * ddfdx_o0 + pnormx_denom * ddenom_o0
    dnormx_o1 = pnormx_dfdx * ddfdx_o1 + pnormx_denom * ddenom_o1
    dnormy_c = pnormy_dfdy * ddfdy_c + pnormy_denom * ddenom_c
    dnormy_o0 = pnormy_dfdy * ddfdy_o0 + pnormy_denom * ddenom_o0
    dnormy_o1 = pnormy_dfdy * ddfdy_o1 + pnormy_denom * ddenom_o1
    dnormz_c, dnormz_o0, dnormz_o1 = pnormz_denom * ddenom_c, pnormz_denom * ddenom_o0, pnormz_denom * ddenom_o1

    if forward:
        dnormx_c, dnormx_o0, dnormx_o1 = -dnormx_c, -dnormx_o0, -dnormx_o1
        dnormy_c, dnormy_o0, dnormy_o1 = -dnormy_c, -dnormy_o0, -dnormy_o1
        dnormz_c, dnormz_o0, dnormz_o1 = -dnormz_c, -dnormz_o0, -dnormz_o1
        pnormx_dsdr2_1, pnormy_dsdr2_1, pnormz_dsdr2_1 = -pnormx_dsdr2_1, -pnormy_dsdr2_1, -pnormz_dsdr2_1

    pcosi_normx, pcosi_normy, pcosi_normz = B_d0, B_d1, B_d2
    pcosi_dsdr2_1 = pcosi_normx * pnormx_dsdr2_1 + pcosi_normy * pnormy_dsdr2_1 + pcosi_normz * pnormz_dsdr2_1
    dcosi_d0, dcosi_d1, dcosi_d2 = norm_x, norm_y, norm_z
    dcosi_c = pcosi_normx * dnormx_c + pcosi_normy * dnormy_c + pcosi_normz * dnormz_c
    dcosi_o0 = pcosi_normx * dnormx_o0 + pcosi_normy * dnormy_o0 + pcosi_normz * dnormz_o0
    dcosi_o1 = pcosi_normx * dnormx_o1 + pcosi_normy * dnormy_o1 + pcosi_normz * dnormz_o1

    psr_cosi = eta * eta * valid_tir * cosi / sr
    psr_dsdr2_1 = psr_cosi * pcosi_dsdr2_1
    dsr_c, dsr_o0, dsr_o1, dsr_d0, dsr_d1, dsr_d2 = (
        psr_cosi * dcosi_c,
        psr_cosi * dcosi_o0,
        psr_cosi * dcosi_o1,
        psr_cosi * dcosi_d0,
        psr_cosi * dcosi_d1,
        psr_cosi * dcosi_d2,
    )
    pd0_sr, pd0_normx, pd0_cosi = norm_x, sr - eta * cosi, -eta * norm_x
    pd0_dsdr2_1 = pd0_cosi * pcosi_dsdr2_1 + pd0_sr * psr_dsdr2_1 + pd0_normx * pnormx_dsdr2_1
    dd0_c = pd0_cosi * dcosi_c + pd0_normx * dnormx_c + pd0_sr * dsr_c
    dd0_o0 = pd0_cosi * dcosi_o0 + pd0_normx * dnormx_o0 + pd0_sr * dsr_o0
    dd0_o1 = pd0_cosi * dcosi_o1 + pd0_normx * dnormx_o1 + pd0_sr * dsr_o1
    dd0_d0 = pd0_cosi * dcosi_d0 + pd0_sr * dsr_d0 + eta
    dd0_d1 = pd0_cosi * dcosi_d1 + pd0_sr * dsr_d1
    dd0_d2 = pd0_cosi * dcosi_d2 + pd0_sr * dsr_d2

    pd1_sr, pd1_normy, pd1_cosi = norm_y, sr - eta * cosi, -eta * norm_y
    pd1_dsdr2_1 = pd1_cosi * pcosi_dsdr2_1 + pd1_sr * psr_dsdr2_1 + pd1_normy * pnormy_dsdr2_1
    dd1_c = pd1_cosi * dcosi_c + pd1_normy * dnormy_c + pd1_sr * dsr_c
    dd1_o0 = pd1_cosi * dcosi_o0 + pd1_normy * dnormy_o0 + pd1_sr * dsr_o0
    dd1_o1 = pd1_cosi * dcosi_o1 + pd1_normy * dnormy_o1 + pd1_sr * dsr_o1
    dd1_d0 = pd1_cosi * dcosi_d0 + pd1_sr * dsr_d0
    dd1_d1 = pd1_cosi * dcosi_d1 + pd1_sr * dsr_d1 + eta
    dd1_d2 = pd1_cosi * dcosi_d2 + pd1_sr * dsr_d2

    pd2_sr, pd2_normz, pd2_cosi = norm_z, sr - eta * cosi, -eta * norm_z
    pd2_dsdr2_1 = pd2_cosi * pcosi_dsdr2_1 + pd2_sr * psr_dsdr2_1 + pd2_normz * pnormz_dsdr2_1
    dd2_c = pd2_cosi * dcosi_c + pd2_normz * dnormz_c + pd2_sr * dsr_c
    dd2_o0 = pd2_cosi * dcosi_o0 + pd2_normz * dnormz_o0 + pd2_sr * dsr_o0
    dd2_o1 = pd2_cosi * dcosi_o1 + pd2_normz * dnormz_o1 + pd2_sr * dsr_o1
    dd2_d0 = pd2_cosi * dcosi_d0 + pd2_sr * dsr_d0
    dd2_d1 = pd2_cosi * dcosi_d1 + pd2_sr * dsr_d1
    dd2_d2 = pd2_cosi * dcosi_d2 + pd2_sr * dsr_d2 + eta

    pnewd0_d0, pnewd0_Bd0 = tl.where(valid_tir, 1, 0), tl.where(valid_tir, 0, 1)
    pnewd0_dsdr2_1 = pnewd0_d0 * pd0_dsdr2_1
    dnewd0_c = pnewd0_d0 * dd0_c
    dnewd0_o0 = pnewd0_d0 * dd0_o0
    dnewd0_o1 = pnewd0_d0 * dd0_o1
    dnewd0_d0 = pnewd0_d0 * dd0_d0 + pnewd0_Bd0
    dnewd0_d1 = pnewd0_d0 * dd0_d1
    dnewd0_d2 = pnewd0_d0 * dd0_d2

    pnewd1_d1, pnewd1_Bd1 = tl.where(valid_tir, 1, 0), tl.where(valid_tir, 0, 1)
    pnewd1_dsdr2_1 = pnewd1_d1 * pd1_dsdr2_1
    dnewd1_c = pnewd1_d1 * dd1_c
    dnewd1_o0 = pnewd1_d1 * dd1_o0
    dnewd1_o1 = pnewd1_d1 * dd1_o1
    dnewd1_d0 = pnewd1_d1 * dd1_d0
    dnewd1_d1 = pnewd1_d1 * dd1_d1 + pnewd1_Bd1
    dnewd1_d2 = pnewd1_d1 * dd1_d2

    pnewd2_d2, pnewd2_Bd2 = tl.where(valid_tir, 1, 0), tl.where(valid_tir, 0, 1)
    pnewd2_dsdr2_1 = pnewd2_d2 * pd2_dsdr2_1
    dnewd2_c = pnewd2_d2 * dd2_c
    dnewd2_o0 = pnewd2_d2 * dd2_o0
    dnewd2_o1 = pnewd2_d2 * dd2_o1
    dnewd2_d0 = pnewd2_d2 * dd2_d0
    dnewd2_d1 = pnewd2_d2 * dd2_d1
    dnewd2_d2 = pnewd2_d2 * dd2_d2 + pnewd2_Bd2

    pobliq_newd0, pobliq_newd1, pobliq_newd2 = B_d0, B_d1, B_d2
    pobliq_B_d0, pobliq_B_d1, pobliq_B_d2 = new_d0, new_d1, new_d2
    dobliq_dsdr2_1 = pobliq_newd0 * pnewd0_dsdr2_1 + pobliq_newd1 * pnewd1_dsdr2_1 + pobliq_newd2 * pnewd2_dsdr2_1
    dobliq_c = pobliq_newd0 * dnewd0_c + pobliq_newd1 * dnewd1_c + pobliq_newd2 * dnewd2_c
    dobliq_o0 = pobliq_newd0 * dnewd0_o0 + pobliq_newd1 * dnewd1_o0 + pobliq_newd2 * dnewd2_o0
    dobliq_o1 = pobliq_newd0 * dnewd0_o1 + pobliq_newd1 * dnewd1_o1 + pobliq_newd2 * dnewd2_o1
    dobliq_d0 = pobliq_newd0 * dnewd0_d0 + pobliq_newd1 * dnewd1_d0 + pobliq_newd2 * dnewd2_d0 + pobliq_B_d0
    dobliq_d1 = pobliq_newd0 * dnewd0_d1 + pobliq_newd1 * dnewd1_d1 + pobliq_newd2 * dnewd2_d1 + pobliq_B_d1
    dobliq_d2 = pobliq_newd0 * dnewd0_d2 + pobliq_newd1 * dnewd1_d2 + pobliq_newd2 * dnewd2_d2 + pobliq_B_d2

    pnewobliq_obliq, pnewobliq_B_obliq = tl.where(valid_tir, 1, 0), tl.where(valid_tir, 0, 1)
    d_obliq = B_dobliq * pnewobliq_B_obliq

    dnewobliq_dsdr2_1 = pnewobliq_obliq * dobliq_dsdr2_1
    dnewobliq_c = pnewobliq_obliq * dobliq_c
    dnewobliq_o0 = pnewobliq_obliq * dobliq_o0
    dnewobliq_o1 = pnewobliq_obliq * dobliq_o1
    dnewobliq_d0 = pnewobliq_obliq * dobliq_d0
    dnewobliq_d1 = pnewobliq_obliq * dobliq_d1
    dnewobliq_d2 = pnewobliq_obliq * dobliq_d2

    d_c = B_dd0 * dnewd0_c + B_dd1 * dnewd1_c + B_dd2 * dnewd2_c + B_dobliq * dnewobliq_c
    d_o0 = B_dd0 * dnewd0_o0 + B_dd1 * dnewd1_o0 + B_dd2 * dnewd2_o0 + B_dobliq * dnewobliq_o0
    d_o1 = B_dd0 * dnewd0_o1 + B_dd1 * dnewd1_o1 + B_dd2 * dnewd2_o1 + B_dobliq * dnewobliq_o1
    d_d0 = B_dd0 * dnewd0_d0 + B_dd1 * dnewd1_d0 + B_dd2 * dnewd2_d0 + B_dobliq * dnewobliq_d0
    d_d1 = B_dd0 * dnewd0_d1 + B_dd1 * dnewd1_d1 + B_dd2 * dnewd2_d1 + B_dobliq * dnewobliq_d1
    d_d2 = B_dd0 * dnewd0_d2 + B_dd1 * dnewd1_d2 + B_dd2 * dnewd2_d2 + B_dobliq * dnewobliq_d2

    d_dsdr2_1 = B_dd0 * pnewd0_dsdr2_1 + B_dd1 * pnewd1_dsdr2_1 + B_dd2 * pnewd2_dsdr2_1 + B_dobliq * dnewobliq_dsdr2_1
    d_ra = B_dra * valid_tir
    # dd_c = B_dd0 * dd0_c + B_dd1 * dd1_c + B_dd2 * dd2_c
    # pd_dsdr2_1 = B_dd0 * pd0_dsdr2_1 + B_dd1 * pd1_dsdr2_1 + B_dd2 * pd2_dsdr2_1
    # dd_o0 = B_dd0 * dd0_o0 + B_dd1 * dd1_o0 + B_dd2 * dd2_o0
    # dd_o1 = B_dd0 * dd0_o1 + B_dd1 * dd1_o1 + B_dd2 * dd2_o1
    # dd_d0 = B_dd0 * dd0_d0 + B_dd1 * dd1_d0 + B_dd2 * dd2_d0
    # dd_d1 = B_dd0 * dd0_d1 + B_dd1 * dd1_d1 + B_dd2 * dd2_d1
    # dd_d2 = B_dd0 * dd0_d2 + B_dd1 * dd1_d2 + B_dd2 * dd2_d2
    dd_ai2 = d_dsdr2_1
    dd_ai4 = d_dsdr2_1 * 2 * valid_r2
    dd_ai6 = d_dsdr2_1 * 3 * valid_r2_2
    dd_ai8 = d_dsdr2_1 * 4 * valid_r2_3
    dd_ai10 = d_dsdr2_1 * 5 * valid_r2_4
    dd_ai12 = d_dsdr2_1 * 6 * valid_r2_5

    tl.store(O_B_do0_ptr, d_o0, boundary_check=(0, 1))
    tl.store(O_B_do1_ptr, d_o1, boundary_check=(0, 1))
    tl.store(O_B_dd0_ptr, d_d0, boundary_check=(0, 1))
    tl.store(O_B_dd1_ptr, d_d1, boundary_check=(0, 1))
    tl.store(O_B_dd2_ptr, d_d2, boundary_check=(0, 1))
    tl.store(O_B_dra_ptr, d_ra, boundary_check=(0, 1))
    tl.store(O_B_dobliq_ptr, d_obliq, boundary_check=(0, 1))
    dd_c = tl.where(mask, d_c, 0)
    dd_ai2 = tl.where(mask, dd_ai2, 0)
    dd_ai4 = tl.where(mask, dd_ai4, 0)
    dd_ai6 = tl.where(mask, dd_ai6, 0)
    dd_ai8 = tl.where(mask, dd_ai8, 0)
    dd_ai10 = tl.where(mask, dd_ai10, 0)
    dd_ai12 = tl.where(mask, dd_ai12, 0)
    while tl.atomic_cas(LOCK, 0, 1) == 1:
        pass
    dd_c_sum = tl.sum(dd_c)
    dd_ai2_sum = tl.sum(dd_ai2)
    dd_ai4_sum = tl.sum(dd_ai4)
    dd_ai6_sum = tl.sum(dd_ai6)
    dd_ai8_sum = tl.sum(dd_ai8)
    dd_ai10_sum = tl.sum(dd_ai10)
    dd_ai12_sum = tl.sum(dd_ai12)
    tl.store(o_ai2_ptr + pid, dd_ai2_sum)
    tl.store(o_ai4_ptr + pid, dd_ai4_sum)
    tl.store(o_ai6_ptr + pid, dd_ai6_sum)
    tl.store(o_ai8_ptr + pid, dd_ai8_sum)
    tl.store(o_ai10_ptr + pid, dd_ai10_sum)
    tl.store(o_ai12_ptr + pid, dd_ai12_sum)
    tl.store(o_c_ptr + pid, dd_c_sum)
    tl.atomic_xchg(LOCK, 0)


def flash_refaction_bwd(i_o, i_d, i_ra, i_obliq, d_d, d_ra, d_obliq, eta, d, k, c, ai2, ai4, ai6, ai8, ai10, ai12):

    # B, G1, G2, D = i_o.shape
    input_shape = i_o.shape
    i_o, i_d, i_ra, i_obliq = i_o.reshape(-1, input_shape[-1]), i_d.reshape(-1, input_shape[-1]), i_ra.reshape(-1, 1), i_obliq.reshape(-1, 1)
    d_d, d_ra, d_obliq = d_d.reshape(-1, input_shape[-1]), d_ra.reshape(-1, 1), d_obliq.reshape(-1, 1)
    (T, D), (C,) = i_o.shape, eta.shape
    o_do, o_dd, o_dra, o_dobliq = torch.zeros_like(i_o), torch.zeros_like(i_d), torch.zeros_like(i_ra), torch.zeros_like(i_obliq)

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
    # fmt:off
    flash_refaction_bwd_kernel[grid](
        i_o, i_d,i_ra,i_obliq, d_d, d_ra,d_obliq,eta, d,k, c,
        ai2, ai4, ai6, ai8, ai10, ai12, locks,
        o_do, o_dd, o_dra,o_dobliq,o_d, o_k, o_c, o_ai2, o_ai4,  o_ai6, o_ai8, o_ai10, o_ai12,
        C,T,D,BS=BLOCK_SIZE,
    )
    # fmt: on
    o_do, o_dd = o_do.reshape(input_shape), o_dd.reshape(input_shape)
    o_dra, o_dobliq = o_dra.reshape(input_shape[:-1]), o_dobliq.reshape(input_shape[:-1])
    return o_do, o_dd, o_dra, o_dobliq, o_d, o_k, o_c, o_ai2, o_ai4, o_ai6, o_ai8, o_ai10, o_ai12


class FlashRefractionMethodFunction(torch.autograd.Function):
    @staticmethod
    @contiguous
    @autocast_custom_fwd
    def forward(ctx, i_o, i_d, i_ra, i_obliq, eta, d, k, c, ai2, ai4, ai6, ai8, ai10, ai12):

        new_d, new_ra, new_obliq, valid = flash_refaction_fwd(i_o, i_d, i_ra, i_obliq, eta, d, k, c, ai2, ai4, ai6, ai8, ai10, ai12)
        ctx.save_for_backward(i_o, i_d, i_ra, i_obliq, eta, d, k, c, ai2, ai4, ai6, ai8, ai10, ai12)
        return new_d, new_ra, new_obliq, valid

    @staticmethod
    @contiguous
    @autocast_custom_bwd
    def backward(ctx, d_d, d_ra, d_obliq, d_valid):
        i_o, i_d, i_ra, i_obliq, eta, d, k, c, ai2, ai4, ai6, ai8, ai10, ai12 = ctx.saved_tensors
        o_do, o_dd, o_dra, o_dobliq, o_d, o_k, o_c, o_ai2, o_ai4, o_ai6, o_ai8, o_ai10, o_ai12 = flash_refaction_bwd(
            i_o, i_d, i_ra, i_obliq, d_d, d_ra, d_obliq, eta, d, k, c, ai2, ai4, ai6, ai8, ai10, ai12
        )

        dd_ai2, dd_ai4, dd_ai6, dd_ai8, dd_ai10, dd_ai12, dd_d, dd_c = (
            torch.sum(o_ai2, 0, keepdim=True),
            torch.sum(o_ai4, 0, keepdim=True),
            torch.sum(o_ai6, 0, keepdim=True),
            torch.sum(o_ai8, 0, keepdim=True),
            torch.sum(o_ai10, 0, keepdim=True),
            torch.sum(o_ai12, 0, keepdim=True),
            torch.sum(o_d, 0, keepdim=True),
            torch.sum(o_c, 0, keepdim=True),
        )
        return o_do, o_dd, o_dra, o_dobliq, None, dd_d, None, dd_c, dd_ai2, dd_ai4, dd_ai6, dd_ai8, dd_ai10, dd_ai12


@torch.compiler.disable
def flash_refaction(i_o, i_d, i_ra, i_obliq, eta, d, k, c, ai2, ai4, ai6, ai8, ai10, ai12):
    eta = eta.to(device=i_o.device)
    new_d, new_ra, new_obliq, valid = FlashRefractionMethodFunction.apply(i_o, i_d, i_ra, i_obliq, eta, d, k, c, ai2, ai4, ai6, ai8, ai10, ai12)
    return new_d, new_ra, new_obliq, valid
