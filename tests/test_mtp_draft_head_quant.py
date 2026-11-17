# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""The fp8 draft head's quantization math, buffer reuse and install gates."""

from __future__ import annotations

import types
import unittest
from unittest import mock

import torch

from hcu_sglang.models import mtp_lmhead_quant as mtp_head
from hcu_sglang.models import qwen3_5_mtp_lmhead_quant as head_install


class QuantizationTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_rows_are_scaled_per_row_and_reconstruct(self):
        weight = torch.randn(5, 4, dtype=torch.bfloat16) * 3
        weight_q, scale = mtp_head.SCHEME.quantize_rows(weight)
        self.assertEqual(weight_q.dtype, mtp_head.SCHEME.dtype)
        self.assertEqual(scale.dtype, torch.float32)
        self.assertEqual(tuple(scale.shape), (5, 1))
        error = (weight_q.float() * scale - weight.float()).abs()
        # e4m3 keeps 3 mantissa bits: the relative error is bounded by 2**-4.
        row_amax = weight.float().abs().amax(dim=-1, keepdim=True)
        self.assertTrue(torch.all(error <= row_amax * 0.07 + 1e-4))
        row_amax = weight.float().abs().amax(dim=-1, keepdim=True)
        self.assertTrue(torch.allclose(
            scale, row_amax / mtp_head.SCHEME.amax, rtol=1e-3, atol=1e-6))

    def test_activations_use_the_same_scale_rule(self):
        x = torch.randn(3, 4, dtype=torch.bfloat16)
        xq, scale = mtp_head.SCHEME.quantize_activations(x)
        self.assertEqual(xq.dtype, mtp_head.SCHEME.dtype)
        self.assertEqual(tuple(scale.shape), (3, 1))
        error = (xq.float() * scale - x.float()).abs()
        row_amax = x.float().abs().amax(dim=-1, keepdim=True)
        self.assertTrue(torch.all(error <= row_amax * 0.07 + 1e-4))

    def test_zero_rows_do_not_divide_by_zero(self):
        weight = torch.zeros(2, 4, dtype=torch.bfloat16)
        _, scale = mtp_head.SCHEME.quantize_rows(weight)
        self.assertTrue(torch.all(scale > 0))


def _method(max_m=4):
    base = types.SimpleNamespace(
        weight=torch.randn(3, 2, dtype=torch.bfloat16)
    )
    calls = {"count": 0, "out": None}

    def gemm(xq, wq, xs, ws, *, out=None, prepare_shape=True):
        calls["count"] += 1
        calls["out"] = out
        out.fill_(1)
        return out

    method = mtp_head.DraftHeadQuantMethod(
        base=base,
        weight_q=torch.zeros(3, 2, dtype=mtp_head.SCHEME.dtype),
        weight_scale=torch.ones(3, 1),
        gemm=gemm,
        max_m=max_m,
        scheme=mtp_head.SCHEME,
    )
    return method, base, calls


class ApplyTest(unittest.TestCase):
    def test_a_narrow_call_reuses_the_preallocated_row(self):
        method, _, calls = _method(max_m=4)
        x = torch.randn(2, 2, dtype=torch.bfloat16)
        out = method.apply(layer=None, x=x)
        self.assertEqual(out.data_ptr(), method.out.data_ptr())
        self.assertEqual(tuple(out.shape), (2, 3))
        self.assertIs(calls["out"], out)
        self.assertEqual(calls["count"], 1)
        self.assertTrue(torch.all(out == 1))

    def test_wide_rows_take_the_weight_directly(self):
        method, base, calls = _method(max_m=2)
        x = torch.randn(3, 2, dtype=torch.bfloat16)
        out = method.apply(layer=None, x=x)
        self.assertTrue(torch.equal(out, torch.matmul(x, base.weight.t())))
        self.assertEqual(calls["count"], 0)

    def test_a_bias_is_a_programming_error(self):
        method, _, _ = _method()
        with self.assertRaises(AssertionError):
            method.apply(layer=None, x=torch.zeros(1, 2), bias=object())


class InstallGateTest(unittest.TestCase):
    def setUp(self):
        self.model = types.SimpleNamespace(
            lm_head=types.SimpleNamespace(
                weight=torch.zeros(4, 2, dtype=torch.bfloat16)
            )
        )

    def _should(self, env="1", **attrs):
        for name, value in attrs.items():
            setattr(self.model, name, value)
        with mock.patch.dict("os.environ", {"SGLANG_MTP_LMHEAD_FP8": env}, clear=False), \
                mock.patch("torch.cuda.is_current_stream_capturing",
                           lambda: False):
            return head_install._should_use_fp8_draft_head(self.model)

    def test_install_is_opt_in(self):
        self.assertFalse(self._should(env="0"))

    def test_an_installed_head_is_not_wrapped_twice(self):
        self.assertFalse(self._should(_mtp_fp8_installed=True))

    def test_a_captured_stream_is_left_alone(self):
        with mock.patch.dict("os.environ", {"SGLANG_MTP_LMHEAD_FP8": "1"}), \
                mock.patch("torch.cuda.is_current_stream_capturing",
                           lambda: True):
            self.assertFalse(
                head_install._should_use_fp8_draft_head(self.model))

    def test_an_existing_proxy_is_not_rewrapped(self):
        proxy = mtp_head.DraftHeadQuantProxy(object(), object())
        self.assertFalse(self._should(lm_head=proxy))

    def test_a_non_matrix_or_wrong_dtype_head_is_rejected(self):
        self.assertFalse(self._should(
            lm_head=types.SimpleNamespace(weight=object())))
        self.assertFalse(self._should(
            lm_head=types.SimpleNamespace(
                weight=torch.zeros(4, 2, dtype=torch.float32))))

    def test_fp32_logits_keep_the_head_bf16(self):
        self.assertFalse(self._should(
            logits_processor=types.SimpleNamespace(use_fp32_lm_head=True)))

    @unittest.skipUnless(torch.cuda.is_available(), "needs a device")
    def test_the_happy_path_is_a_bf16_cuda_matrix(self):
        self.model.lm_head.weight = torch.zeros(4, 2, dtype=torch.bfloat16,
                                                device="cuda")
        self.assertTrue(self._should())

    def test_the_proxy_forwards_everything_but_its_own_state(self):
        base = types.SimpleNamespace(weight="w", other=7)
        proxy = mtp_head.DraftHeadQuantProxy(base, method="m")
        self.assertEqual(proxy.other, 7)
        self.assertEqual(proxy.weight, "w")
        self.assertEqual(proxy.quant_method, "m")
        for name in ("set_lora", "apply_lora"):
            with self.assertRaises(AttributeError):
                proxy.__getattr__(name)


if __name__ == "__main__":
    unittest.main()
