# Anima Reference for Forge Neo
# The Forge side with stand-ins for the few Forge Neo names it touches: the compat check, the forward and attention
# dispatch, the Ref Adapter CFG function with the branch without the reference, and the Role Ref KV cache.

import sys
import types
import unittest
from unittest import mock

import torch

from lib_anima_ref import adapter as lib_adapter
from lib_anima_ref import guidance, kv, patches
from tests.test_adapter import lora_tensors
from tests.toy_dit import Attention, ToyAnima


class FakeDiT(ToyAnima):
    def forward(self, x, timesteps, context, padding_mask=None, **kwargs):
        return torch.full_like(x, -1.0)


class FakeEngine:
    pass


class FakeUnet:
    def __init__(self):
        self.model_options = {"transformer_options": {}}
        self.model = types.SimpleNamespace(computation_dtype=torch.float16)
        self.preserved = 0

    def clone(self):
        n = FakeUnet()
        n.model_options = {k: (v.copy() if isinstance(v, (dict, list)) else v) for k, v in self.model_options.items()}
        n.preserved = self.preserved
        return n

    def add_extra_preserved_memory_during_sampling(self, n):
        self.preserved += n

    def set_transformer_option(self, k, v):
        self.model_options["transformer_options"][k] = v

    def set_model_sampler_cfg_function(self, fn, disable_cfg1_optimization=False):
        self.model_options["sampler_cfg_function"] = fn
        if disable_cfg1_optimization:
            self.model_options["disable_cfg1_optimization"] = True

    def set_model_sampler_pre_cfg_function(self, fn, disable_cfg1_optimization=False):
        self.model_options["sampler_pre_cfg_function"] = self.model_options.get("sampler_pre_cfg_function", []) + [fn]
        if disable_cfg1_optimization:
            self.model_options["disable_cfg1_optimization"] = True


def forge_modules(calc=None):
    def module(name, **attributes):
        result = types.ModuleType(name)
        result.__dict__.update(attributes)
        return result

    return {
        "backend": module("backend"),
        "backend.nn": module("backend.nn"),
        "backend.nn.anima": module("backend.nn.anima", Anima=FakeDiT, SelfCrossAttention=Attention),
        "backend.diffusion_engine": module("backend.diffusion_engine"),
        "backend.diffusion_engine.anima": module("backend.diffusion_engine.anima", Anima=FakeEngine),
        "backend.sampling": module("backend.sampling"),
        "backend.sampling.sampling_function": module("backend.sampling.sampling_function", calc_cond_uncond_batch=calc or (lambda *a: None)),
        "backend.patcher": module("backend.patcher"),
        "backend.patcher.unet": module("backend.patcher.unet", UnetPatcher=FakeUnet),
    }


class PatchesTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.originals = {FakeDiT: FakeDiT.forward, Attention: Attention.forward}

    def tearDown(self):
        for cls, forward in self.originals.items():
            cls.forward = forward
            if "_anima_ref_original" in cls.__dict__:
                del cls._anima_ref_original

    def test_check_passes_on_the_expected_forge(self):
        with mock.patch.dict(sys.modules, forge_modules()):
            self.assertEqual(patches.check(), [])

    def test_check_reports_a_missing_module(self):
        modules = forge_modules()
        modules["backend.sampling.sampling_function"] = types.ModuleType("backend.sampling.sampling_function")
        with mock.patch.dict(sys.modules, modules):
            problems = patches.check()
        self.assertEqual(len(problems), 1)
        self.assertIn("calc_cond_uncond_batch", problems[0])

    def test_forward_dispatch(self):
        with mock.patch.dict(sys.modules, forge_modules()):
            patches.apply()
            patches.apply()  # twice is once
        self.assertIs(FakeDiT._anima_ref_original, self.originals[FakeDiT])
        dm = FakeDiT().eval()
        x, t, context = torch.randn(1, 4, 1, 8, 6), torch.tensor([500.0]), torch.randn(1, 5, 8)
        self.assertTrue(torch.equal(dm(x, t, context, transformer_options={}), torch.full_like(x, -1.0)))

        adapter = lib_adapter.Adapter(lora_tensors(dm))
        seen = []
        dm.blocks[0].self_attn.q_proj.register_forward_hook(lambda *a: seen.append(lib_adapter._active["adapter"]))
        reference = patches.Reference([torch.randn(1, 4, 1, 8, 6)], adapter, 0.8, 1.0)
        with torch.no_grad():
            out = dm(x, t, context, transformer_options={patches.OPTION_KEY: reference})
        self.assertEqual(out.shape, x.shape)
        self.assertFalse(torch.equal(out, torch.full_like(x, -1.0)))
        self.assertIs(seen[-1], adapter)
        self.assertIsNone(lib_adapter._active["adapter"])

    def test_cfg_adds_the_branch_without_the_reference(self):
        calls = []
        base = torch.randn(1, 4, 1, 6, 6)

        def calc(model, cond, uncond, x, timestep, model_options):
            calls.append((cond, uncond, model_options))
            return torch.zeros_like(x), base

        with mock.patch.dict(sys.modules, forge_modules(calc)):
            reference = types.SimpleNamespace(adapter=lib_adapter.Adapter(lora_tensors(ToyAnima())))
            unet = patches.patch_unet(FakeUnet(), reference, 4.5, "apg", 0.5)
            self.assertGreater(unet.preserved, 0)
            options = unet.model_options
            self.assertIs(options["transformer_options"][patches.OPTION_KEY], reference)
            self.assertTrue(options["disable_cfg1_optimization"])

            x, pos, neg = (torch.randn(1, 4, 1, 6, 6) for _ in range(3))
            options["sampler_pre_cfg_function"][0]("model", ["c"], ["u"], x, "t", options)
            result = options["sampler_cfg_function"]({"input": x, "cond_denoised": pos, "uncond_denoised": neg, "cond_scale": 3.0, "timestep": "t", "model": "model", "model_options": options})

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][:2], ([], ["u"]))
        self.assertNotIn(patches.OPTION_KEY, calls[0][2]["transformer_options"])
        self.assertIn(patches.OPTION_KEY, options["transformer_options"])  # the sampling's own options keep it
        torch.testing.assert_close(x - result, guidance.combine(pos, neg, base, 3.0, 4.5, "apg", 0.5))

    def test_ref_guidance_one_skips_the_extra_branch(self):
        calls = []
        with mock.patch.dict(sys.modules, forge_modules(lambda *a: calls.append(a))):
            unet = patches.patch_unet(FakeUnet(), types.SimpleNamespace(adapter=lib_adapter.Adapter(lora_tensors(ToyAnima()))), 1.0, "none", 0.5)
            options = unet.model_options
            self.assertNotIn("disable_cfg1_optimization", options)
            x, pos, neg = (torch.randn(1, 4, 1, 6, 6) for _ in range(3))
            options["sampler_pre_cfg_function"][0]("model", ["c"], ["u"], x, "t", options)
            result = options["sampler_cfg_function"]({"input": x, "cond_denoised": pos, "uncond_denoised": neg, "cond_scale": 4.0, "timestep": "t", "model": "model", "model_options": options})
        self.assertEqual(calls, [])
        torch.testing.assert_close(x - result, neg + 4.0 * (pos - neg))

    def kv_setup(self):
        with mock.patch.dict(sys.modules, forge_modules()):
            patches.apply()
        dm = FakeDiT().eval()
        adapter = lib_adapter.Adapter(lora_tensors(dm), compute_dtype=torch.float32)
        reference = patches.RoleRefKVReference(torch.randn(1, 4, 1, 6, 4), adapter, 1.0)
        x, t, context = torch.randn(2, 4, 1, 8, 6), torch.tensor([500.0, 500.0]), torch.randn(2, 5, 8)
        return dm, adapter, reference, x, t, context

    def test_attention_dispatch_leaves_plain_calls_alone(self):
        dm, *_ = self.kv_setup()
        attn = dm.blocks[0].self_attn
        x = torch.randn(1, 6, 24)
        torch.testing.assert_close(attn(x, None, transformer_options={}), self.originals[Attention](attn, x, None))
        self.assertIs(Attention._anima_ref_original, self.originals[Attention])

    def test_kv_captures_once_and_only_cond_rows_use_it(self):
        dm, adapter, reference, x, t, context = self.kv_setup()
        seen = []
        dm.blocks[0].self_attn.q_proj.register_forward_hook(lambda *a: seen.append(lib_adapter._active["adapter"]))
        options = {patches.OPTION_KEY: reference, "cond_or_uncond": [0, 1]}
        with torch.no_grad(), mock.patch.object(kv, "capture", wraps=kv.capture) as capture:
            out = dm(x, t, context, transformer_options=options)
            again = dm(x, t, context, transformer_options=options)
            uncond = dm(x[1:], t[1:], context[1:], transformer_options={patches.OPTION_KEY: reference, "cond_or_uncond": [1]})
            with lib_adapter.active(adapter, 1.0), kv.qk_eps(dm):
                no_cache = kv.kv_forward(dm, x[:1], t[:1], context[:1], [], {})
        self.assertEqual(capture.call_count, 1)
        self.assertEqual(len(reference.caches), 1)
        self.assertEqual(len(next(iter(reference.caches.values()))), len(dm.blocks))
        self.assertEqual(out.shape, x.shape)
        torch.testing.assert_close(out, again)
        torch.testing.assert_close(out[1:], uncond)
        self.assertFalse(torch.allclose(out[:1], no_cache))
        self.assertTrue(seen and all(a is adapter for a in seen))  # the LoRA is on in capture, cond and uncond
        self.assertIsNone(lib_adapter._active["adapter"])
        self.assertEqual({b.self_attn.q_norm.eps for b in dm.blocks}, {1e-6})

    def test_kv_negative_only_pass_before_any_positive(self):
        dm, _, reference, x, t, context = self.kv_setup()
        with torch.no_grad():
            out = dm(x[:1], t[:1], context[:1], transformer_options={patches.OPTION_KEY: reference, "cond_or_uncond": [1]})
        self.assertEqual(out.shape, x[:1].shape)
        self.assertEqual(reference.caches, {})

    def test_patch_unet_kv_is_plain_cfg(self):
        reference = patches.RoleRefKVReference(torch.zeros(1), lib_adapter.Adapter(lora_tensors(ToyAnima()), compute_dtype=torch.float32), 1.0)
        unet = patches.patch_unet_kv(FakeUnet(), reference)
        self.assertIs(unet.model_options["transformer_options"][patches.OPTION_KEY], reference)
        self.assertNotIn("sampler_cfg_function", unet.model_options)
        self.assertEqual(unet.preserved, reference.adapter.nbytes(torch.float16))


if __name__ == "__main__":
    unittest.main()
