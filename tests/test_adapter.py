# Anima Reference for Forge Neo
# The adapter LoRA: weights read like the released file, hooks that only act inside `active`, metadata detection.

import json
import os
import tempfile
import unittest

import torch
from safetensors.torch import save_file

from lib_anima_ref import adapter as lib_adapter
from lib_anima_ref import files
from lib_anima_ref.adapter import Adapter, active
from tests.toy_dit import ToyAnima

RANK = 2


def lora_tensors(dm, targets=("blocks.0.self_attn.q_proj", "blocks.1.mlp.layer2"), alpha=4.0):
    tensors = {}
    for name in targets:
        module = dm.get_submodule(name)
        tensors[f"diffusion_model.{name}.lora_down.weight"] = torch.randn(RANK, module.in_features)
        tensors[f"diffusion_model.{name}.lora_up.weight"] = torch.randn(module.out_features, RANK)
        tensors[f"diffusion_model.{name}.alpha"] = torch.tensor(alpha)
    return tensors


class AdapterTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.dm = ToyAnima()
        self.tensors = lora_tensors(self.dm)
        self.adapter = Adapter(self.tensors)

    def test_weights_drop_the_prefix_and_scale_by_alpha_over_rank(self):
        self.assertEqual(sorted(self.adapter.weights), ["blocks.0.self_attn.q_proj", "blocks.1.mlp.layer2"])
        self.assertEqual(self.adapter.weights["blocks.0.self_attn.q_proj"][2], 4.0 / RANK)

    def test_hooks_apply_only_while_active(self):
        self.assertEqual(self.adapter.install(self.dm), 2)
        self.assertEqual(self.adapter.install(self.dm), 0)
        module = self.dm.get_submodule("blocks.0.self_attn.q_proj")
        x = torch.randn(3, module.in_features)
        plain = module(x)
        down = self.tensors["diffusion_model.blocks.0.self_attn.q_proj.lora_down.weight"]
        up = self.tensors["diffusion_model.blocks.0.self_attn.q_proj.lora_up.weight"]
        with active(self.adapter, 0.5):
            torch.testing.assert_close(module(x), plain + (x @ down.T @ up.T) * (4.0 / RANK) * 0.5)
        torch.testing.assert_close(module(x), plain)
        self.assertIsNone(lib_adapter._active["adapter"])

    def test_untargeted_modules_are_untouched(self):
        self.adapter.install(self.dm)
        module = self.dm.get_submodule("blocks.0.self_attn.k_proj")
        x = torch.randn(2, module.in_features)
        with active(self.adapter, 1.0):
            torch.testing.assert_close(module(x), x @ module.weight.T)

    def test_weights_follow_the_input_dtype(self):
        self.adapter.install(self.dm)
        self.dm.to(torch.bfloat16)
        module = self.dm.get_submodule("blocks.1.mlp.layer2")
        with active(self.adapter, 1.0):
            out = module(torch.randn(2, module.in_features, dtype=torch.bfloat16))
        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertEqual(self.adapter.weights["blocks.1.mlp.layer2"][0].dtype, torch.bfloat16)

    def test_missing_alpha_means_rank(self):
        tensors = {k: v for k, v in self.tensors.items() if not k.endswith("alpha")}
        self.assertEqual(Adapter(tensors).weights["blocks.0.self_attn.q_proj"][2], 1.0)

    def test_file_without_lora_weights_is_rejected(self):
        with self.assertRaises(ValueError):
            Adapter({"x": torch.zeros(1)})


    def test_fp32_compute_keeps_the_weights_in_fp32(self):
        adapter = Adapter(self.tensors, compute_dtype=torch.float32)
        adapter.install(self.dm)
        self.dm.to(torch.bfloat16)
        module = self.dm.get_submodule("blocks.0.self_attn.q_proj")
        x = torch.randn(2, module.in_features, dtype=torch.bfloat16)
        with active(adapter, 1.0):
            out = module(x)
        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertEqual(adapter.weights["blocks.0.self_attn.q_proj"][0].dtype, torch.float32)
        self.assertEqual(adapter.nbytes(torch.float16), sum((d.numel() + u.numel()) * 4 for d, u, _ in adapter.weights.values()))
        self.assertEqual(self.adapter.nbytes(torch.float16) * 2, adapter.nbytes(torch.float16))


class MetadataTest(unittest.TestCase):
    def test_each_method_finds_its_files(self):
        meta = {"rank": 32, "alpha": 32, "ref_offset_h": 1.0}
        empty = {"a": torch.zeros(1)}
        with tempfile.TemporaryDirectory() as root:
            os.makedirs(os.path.join(root, "anima"))
            tagged = os.path.join(root, "anima", "anima_v2_pilot.safetensors")
            save_file(lora_tensors(ToyAnima()), tagged, metadata={lib_adapter.REF_ADAPTER_KEY: json.dumps(meta)})
            save_file(empty, os.path.join(root, "anima_style.safetensors"))
            save_file(empty, os.path.join(root, "anima_ref_adapter_copy.safetensors"))
            save_file(empty, os.path.join(root, "other.safetensors"), metadata={lib_adapter.REF_ADAPTER_KEY: "{}"})
            save_file(empty, os.path.join(root, "anima-role-ref-kv-v1.0.safetensors"), metadata={lib_adapter.ROLE_REF_KV_KEY: "1.0"})
            save_file(empty, os.path.join(root, "anima_charref_renamed.safetensors"), metadata={lib_adapter.ROLE_REF_KV_KEY: "1.0"})
            open(os.path.join(root, "notes.txt"), "w").close()

            self.assertEqual(lib_adapter.read_metadata(tagged), meta)
            self.assertIsNone(lib_adapter.read_metadata(os.path.join(root, "anima_style.safetensors")))
            self.assertIsNone(lib_adapter.read_header_metadata(os.path.join(root, "notes.txt")))
            found = files.find_adapters([root, os.path.join(root, "missing")])
            self.assertEqual(list(found[files.REF_ADAPTER]), ["anima/anima_v2_pilot.safetensors", "anima_ref_adapter_copy.safetensors"])
            self.assertEqual(list(found[files.ROLE_REF_KV]), ["anima-role-ref-kv-v1.0.safetensors", "anima_charref_renamed.safetensors"])

            loaded = Adapter.load(tagged)
            self.assertIs(Adapter.load(tagged), loaded)
            self.assertIsNot(Adapter.load(tagged, torch.float32), loaded)
            self.assertEqual(len(loaded.weights), 2)


if __name__ == "__main__":
    unittest.main()
