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


def test_refraction_method():
    lens, ray, _ = init_lens.init()
    clen = lens.surfaces[1]
    mat1 = lens.surfaces[0].mat2
    n1 = mat1.ior(ray.wvln)
    n2 = clen.mat2.ior(ray.wvln)
    tmp_ray1 = ray.clone()
    tmp_ray1.o.requires_grad = True
    tmp_ray1.d.requires_grad = True
    tmp_ray1.ra.requires_grad = True
    tmp_ray1_o, tmp_ray1_d = tmp_ray1.o, tmp_ray1.d
    n2clen = clen.clone()
    ray1 = clen.intersect(tmp_ray1, n1)
    ray1 = clen.refract(ray1, n1 / n2)
    (ray1.o.sum() + ray1.d.sum() + ray1.ra.sum() + ray1.obliq.sum()).backward()
    c_grad, ai2_grad, ai4_grad, ai6_grad, ai8_grad, ai10_grad, ai12_grad = (
        clen.c.grad,
        clen.ai2.grad,
        clen.ai4.grad,
        clen.ai6.grad,
        clen.ai8.grad,
        clen.ai10.grad,
        clen.ai12.grad,
    )


    tmp_ray3 = ray.clone()
    tmp_ray3.o, tmp_ray3.d, tmp_ray3.ra, tmp_ray3.obliq = (
        ray.o.unsqueeze(0).clone(),
        ray.d.unsqueeze(0).clone(),
        ray.ra.unsqueeze(0).clone(),
        ray.obliq.unsqueeze(0).clone(),
    )
    tmp_ray3.o.requires_grad = True
    tmp_ray3.d.requires_grad = True
    tmp_ray3.ra.requires_grad = True
    tmp_ray3.obliq.requires_grad = True
    tmp_ray3_o, tmp_ray3_d = tmp_ray3.o,tmp_ray3.d
    ray3 = n2clen.intersect_multi_waves(tmp_ray3)
    ray3 = n2clen.refract_multi_waves(ray3)
    (ray3.o.sum() + ray3.d.sum() + ray3.ra.sum() + ray3.obliq.sum()).backward()
    n2c_grad, n2ai2_grad, n2ai4_grad, n2ai6_grad, n2ai8_grad, n2ai10_grad, n2ai12_grad = (
        n2clen.c.grad,
        n2clen.ai2.grad,
        n2clen.ai4.grad,
        n2clen.ai6.grad,
        n2clen.ai8.grad,
        n2clen.ai10.grad,
        n2clen.ai12.grad,
    )
    assert_close(ray1.o, ray3.o.squeeze(0))
    assert_close(ray1.d, ray3.d.squeeze(0))
    assert_close(ray1.ra, ray3.ra.squeeze(0))
    assert_close(ray1.obliq, ray3.obliq.squeeze(0))
    assert_close(tmp_ray1_o.grad, tmp_ray3_o.grad.squeeze(0))
    assert_close(tmp_ray1_d.grad, tmp_ray3_d.grad.squeeze(0))
    assert_close(c_grad, n2c_grad)
    assert_close(ai2_grad, n2ai2_grad)
    assert_close(ai4_grad, n2ai4_grad)
    assert_close(ai6_grad, n2ai6_grad)
    assert_close(ai8_grad, n2ai8_grad)
    assert_close(ai10_grad, n2ai10_grad)
    assert_close(ai12_grad, n2ai12_grad)
