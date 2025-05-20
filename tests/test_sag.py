import logging
import os
import sys

import pytest
import torch
from torch.testing import *

sys.path.append(".")
sys.path.append("..")
root_path = os.path.dirname(os.path.dirname(__file__))
os.chdir(os.path.dirname(os.path.dirname(__file__)))
import init_lens


def test_sag():
    lens, ray, _ = init_lens.init()
    clen = lens.surfaces[1]
    mat1 = lens.surfaces[0].mat2
    n1 = mat1.ior(ray.wvln)
    n2 = clen.mat2.ior(ray.wvln)
    # ray = clen.intersect(ray, n1)
    nclen = clen.clone()
    intersect1 = torch.tensor([0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]).to(ray.o.device)
    intersect2 = torch.tensor([0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]).to(ray.o.device)
    zero_y1 = torch.zeros_like(intersect1)
    zero_y2 = torch.zeros_like(intersect2)
    intersect1.requires_grad = True
    zero_y1.requires_grad = True
    z = clen.surface(intersect1, zero_y1, use_flash_method=False)
    z.sum().backward()
    c_grad, ai2_grad, ai4_grad, ai6_grad, ai8_grad, ai10_grad, ai12_grad = (
        clen.c.grad,
        clen.ai2.grad,
        clen.ai4.grad,
        clen.ai6.grad,
        clen.ai8.grad,
        clen.ai10.grad,
        clen.ai12.grad,
    )

    intersect2.requires_grad = True
    zero_y2.requires_grad = True
    z2 = nclen.surface(intersect2, zero_y2, use_flash_method=True)
    z2.sum().backward()
    nc_grad, nai2_grad, nai4_grad, nai6_grad, nai8_grad, nai10_grad, nai12_grad = (
        nclen.c.grad,
        nclen.ai2.grad,
        nclen.ai4.grad,
        nclen.ai6.grad,
        nclen.ai8.grad,
        nclen.ai10.grad,
        nclen.ai12.grad,
    )

    assert_close(z, z2)
    assert_close(intersect1.grad, intersect2.grad)
    assert_close(zero_y1.grad, zero_y2.grad)
    assert_close(c_grad, nc_grad)
    assert_close(ai2_grad, nai2_grad)
    assert_close(ai4_grad, nai4_grad)
    assert_close(ai6_grad, nai6_grad)
    assert_close(ai8_grad, nai8_grad)
    assert_close(ai10_grad, nai10_grad)
    assert_close(ai12_grad, nai12_grad)
