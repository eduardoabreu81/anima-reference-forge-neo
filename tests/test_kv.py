# Anima Reference for Forge Neo
# Role Ref KV pieces that need no Forge: reference size, positions, rows, the q/k eps, and the cache capture.

import unittest

import torch
from PIL import Image

from lib_anima_ref import kv
from tests.toy_dit import ToyAnima


class ReferenceTest(unittest.TestCase):
    def test_resize_caps_the_area_and_rounds_to_16(self):
        big = kv.resize_reference(Image.new("RGB", (2000, 1000)))
        self.assertEqual(big.size, (1072, 528))
        self.assertLessEqual(big.width * big.height, kv.REF_AREA)
        small = kv.resize_reference(Image.new("RGBA", (500, 300)))
        self.assertEqual((small.size, small.mode), ((496, 288), "RGB"))

    def test_reference_grid_is_centred_on_the_target(self):
        hs, ws = kv.reference_positions((48, 32), (36, 36), "cpu")
        self.assertEqual((hs[0].item(), hs[-1].item()), (6, 41))
        self.assertEqual((ws[0].item(), ws[-1].item()), (-2, 33))

    def test_cond_rows_follow_the_chunks(self):
        self.assertEqual(kv.cond_rows({"cond_or_uncond": [0, 1]}, 4), [True, True, False, False])
        self.assertEqual(kv.cond_rows({"cond_or_uncond": [1]}, 2), [False, False])
        self.assertEqual(kv.cond_rows({}, 3), [True, True, True])

    def test_qk_eps_is_set_and_restored(self):
        dm = ToyAnima()
        with kv.qk_eps(dm):
            self.assertEqual({b.self_attn.q_norm.eps for b in dm.blocks} | {b.cross_attn.k_norm.eps for b in dm.blocks}, {kv.QK_EPS})
        self.assertEqual({b.self_attn.k_norm.eps for b in dm.blocks}, {1e-6})


class SelfAttentionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.attn = ToyAnima().blocks[0].self_attn
        self.x = torch.randn(3, 10, 24)
        self.cache = [(torch.randn(1, 7, 2, 12), torch.randn(1, 7, 2, 12))]

    def test_capture_keeps_k_and_v_and_attends_normally(self):
        state = kv.State("capture")
        out = kv.self_attention(self.attn, self.x, None, {}, state)
        torch.testing.assert_close(out, self.attn(self.x))
        _, k, _ = self.attn.compute_qkv(self.x)
        torch.testing.assert_close(state.cache[0][0], k)
        self.assertEqual(state.index, 1)

    def test_only_cond_rows_see_the_reference(self):
        rows = [True, False, True]
        out = kv.self_attention(self.attn, self.x, None, {}, kv.State("inject", self.cache, rows))
        torch.testing.assert_close(out[1], self.attn(self.x[1:2])[0])
        q, k, v = self.attn.compute_qkv(self.x[:1])
        expected = self.attn.compute_attention(q, torch.cat([self.cache[0][0], k], 1), torch.cat([self.cache[0][1], v], 1))
        torch.testing.assert_close(out[0], expected[0])
        self.assertFalse(torch.allclose(out[0], self.attn(self.x[:1])[0]))

    def test_no_cache_entry_means_plain_attention(self):
        out = kv.self_attention(self.attn, self.x, None, {}, kv.State("inject", [], [True] * 3))
        torch.testing.assert_close(out, self.attn(self.x))


if __name__ == "__main__":
    unittest.main()
