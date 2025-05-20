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


def test_newtons_method():
    lens, ray, _ = init_lens.init()
    clen = lens.surfaces[1]
    tmp_ray1 = ray.clone()
    tmp_ray1.o.requires_grad = True
    tmp_ray1.d.requires_grad = True
    tmp_ray1.ra.requires_grad = True
    tmp_ray2 = ray.clone()
    tmp_ray2.o.requires_grad = True
    tmp_ray2.d.requires_grad = True
    tmp_ray2.ra.requires_grad = True
    nclen = clen.clone()
    ray1 = clen.intersect(tmp_ray1, 0)
    (ray1.o.sum() + ray1.d.sum() + ray1.ra.sum() + ray1.obliq.sum()).backward()
    d_grad, c_grad, k_grad, ai2_grad, ai4_grad, ai6_grad, ai8_grad, ai10_grad, ai12_grad = (
        clen.d.grad,
        clen.c.grad,
        clen.k.grad,
        clen.ai2.grad,
        clen.ai4.grad,
        clen.ai6.grad,
        clen.ai8.grad,
        clen.ai10.grad,
        clen.ai12.grad,
    )

    ray2 = nclen.intersect_multi_waves(tmp_ray2)
    (ray2.o.sum() + ray2.d.sum() + ray2.ra.sum() + ray2.obliq.sum()).backward()
    nd_grad, nc_grad, nai2_grad, nai4_grad, nai6_grad, nai8_grad, nai10_grad, nai12_grad = (
        nclen.d.grad,
        nclen.c.grad,
        nclen.ai2.grad,
        nclen.ai4.grad,
        nclen.ai6.grad,
        nclen.ai8.grad,
        nclen.ai10.grad,
        nclen.ai12.grad,
    )
    assert_close(ray1.o, ray2.o)
    assert_close(ray1.d, ray2.d)
    assert_close(ray1.ra, ray2.ra)
    assert_close(ray1.obliq, ray2.obliq)
    assert_close(d_grad, nd_grad)
    assert_close(c_grad, nc_grad)
    assert_close(ai2_grad, nai2_grad)
    assert_close(ai4_grad, nai4_grad)
    assert_close(ai6_grad, nai6_grad)
    assert_close(ai8_grad, nai8_grad)
    assert_close(ai10_grad, nai10_grad)
    assert_close(ai12_grad, nai12_grad)
