# Anima Reference for Forge Neo
# The adapter tags taken out of Forge's LoRA list, with Forge's weight rules.

import os
import unittest

from lib_anima_ref import tags


class Params:
    """modules.extra_networks.ExtraNetworkParams' shape: positional items and named ones."""

    def __init__(self, *positional, **named):
        self.items = list(positional) + [f"{k}={v}" for k, v in named.items()]
        self.positional, self.named = list(positional), named


class TagsTest(unittest.TestCase):
    def test_names_are_the_file_and_its_path_under_a_lora_folder(self):
        root = os.path.join("x", "Lora")
        path = os.path.join(root, "Anima", "anima_ref_adapter_pilot_r32.safetensors")
        self.assertEqual(tags.lora_names(path, [root, None, os.path.join("y", "Other")]), {"anima_ref_adapter_pilot_r32", "anima/anima_ref_adapter_pilot_r32"})

    def test_weights_follow_forge(self):
        self.assertEqual(tags.multiplier(["n"], {}), 1.0)
        self.assertEqual(tags.multiplier(["n", "0.7"], {}), 0.7)
        self.assertEqual(tags.multiplier(["n", "0.7", "0.4"], {}), 0.4)
        self.assertEqual(tags.multiplier(["n", "0.7"], {"unet": "0.2"}), 0.2)

    def test_split_keeps_ordinary_loras_for_forge(self):
        adapter = os.path.join("Lora", "anima-role-ref-kv-v1.0.safetensors")
        params = [Params("detailer", "0.3"), Params("anima-role-ref-kv-v1.0", "0.8"), Params("slider", "0.6")]
        keep, found = tags.split(params, lambda name: adapter if name == "anima-role-ref-kv-v1.0" else None)
        self.assertEqual([p.positional[0] for p in keep], ["detailer", "slider"])
        self.assertEqual(found, [("anima-role-ref-kv-v1.0", adapter, 0.8)])
        self.assertEqual(tags.split([], lambda name: None), ([], []))


if __name__ == "__main__":
    unittest.main()
