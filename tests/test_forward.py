# Anima Reference for Forge Neo
# The two-frame forward pass: RoPE positions, the clean reference frame, padding, batch and reference picking.

import unittest

import torch

from lib_anima_ref import forward
from tests.toy_dit import ToyAnima, VideoRopePosition3DEmb


class RopeTest(unittest.TestCase):
    def test_unshifted_frames_match_the_embedder(self):
        pe = VideoRopePosition3DEmb(12)
        hp, wp = 3, 5
        positions = [forward.frame_positions(0, hp, wp, 0, 0, "cpu"), forward.frame_positions(1, hp, wp, 0, 0, "cpu")]
        rope = forward.rope_from_positions(pe, *(torch.cat(axis) for axis in zip(*positions)))
        expected = pe(torch.zeros(1, 2, hp, wp, 1), device="cpu")
        self.assertEqual(rope.shape, expected.shape)
        torch.testing.assert_close(rope, expected)

    def test_reference_frame_is_shifted_down_by_target_heights(self):
        t, h, w = forward.frame_positions(1, 4, 3, round(1.0 * 4), 0, "cpu")
        self.assertTrue(torch.equal(t, torch.ones(12, dtype=t.dtype)))
        self.assertEqual(h.min().item(), 4)
        self.assertEqual(h.max().item(), 7)
        self.assertEqual(w.max().item(), 2)


class PickRefTest(unittest.TestCase):
    def test_exact_size_wins_and_4d_gets_a_frame_axis(self):
        small, big = torch.randn(1, 4, 6, 4), torch.randn(1, 4, 1, 12, 8)
        self.assertIs(forward.pick_ref([small, big], (12, 8)), big)
        self.assertEqual(forward.pick_ref([small, big], (6, 4)).shape, (1, 4, 1, 6, 4))

    def test_other_sizes_resize_the_closest(self):
        ref = forward.pick_ref([torch.randn(1, 4, 1, 12, 8)], (10, 6))
        self.assertEqual(ref.shape, (1, 4, 1, 10, 6))


class RefForwardTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.dm = ToyAnima().eval()
        self.context = torch.randn(2, 5, 8)

    def run_forward(self, x, ref, offset=1.0, timesteps=None):
        timesteps = torch.full((x.shape[0],), 600.0) if timesteps is None else timesteps
        with torch.no_grad():
            return forward.ref_forward(self.dm, x, timesteps, self.context[: x.shape[0]], [ref], offset, {"k": 1})

    def test_output_is_the_target_frame_only(self):
        x, ref = torch.randn(2, 4, 1, 8, 6), torch.randn(1, 4, 1, 8, 6)
        out = self.run_forward(x, ref)
        self.assertEqual(out.shape, x.shape)
        call = self.dm.blocks[0].calls[-1]
        self.assertEqual(call["x"], (2, 2, 4, 3, 24))  # two frames of 4x3 patches, batch 2 (the reference expanded)
        self.assertEqual(call["rope"][:3], (1, 2 * 4 * 3, 1))  # rope_emb_L_1_1_D as Anima.forward builds it
        self.assertEqual(call["options"], {"k": 1})

    def test_reference_frame_is_clean(self):
        x, ref = torch.randn(2, 4, 1, 8, 6), torch.randn(1, 4, 1, 8, 6)
        self.run_forward(x, ref, timesteps=torch.tensor([700.0, 300.0]))
        seen = self.dm.t_embedder[0].seen[-1]
        self.assertTrue(torch.equal(seen, torch.tensor([[700.0, 0.0], [300.0, 0.0]])))

    def test_odd_latent_sizes_are_padded_and_cropped(self):
        x, ref = torch.randn(1, 4, 1, 7, 5), torch.randn(1, 4, 1, 7, 5)
        self.assertEqual(self.run_forward(x, ref).shape, x.shape)

    def test_reference_and_offset_change_the_result(self):
        x, ref = torch.randn(1, 4, 1, 8, 6), torch.randn(1, 4, 1, 8, 6)
        out = self.run_forward(x, ref)
        self.assertFalse(torch.allclose(out, self.run_forward(x, torch.randn_like(ref))))
        self.assertFalse(torch.allclose(out, self.run_forward(x, ref, offset=0.0)))
        torch.testing.assert_close(out, self.run_forward(x, ref))

    def test_fp16_residual_stream_runs_in_fp32(self):
        dm = self.dm
        seen = []
        dm.blocks[0].register_forward_pre_hook(lambda m, args: seen.append(args[0].dtype))
        dm.half()
        x, ref = torch.randn(1, 4, 1, 8, 6).half(), torch.randn(1, 4, 1, 8, 6)
        try:
            forward.ref_forward(dm, x, torch.tensor([500.0]), self.context[:1].half(), [ref], 1.0, {})
        except RuntimeError as e:  # CPU builds without some fp16 kernels
            self.skipTest(str(e))
        self.assertEqual(seen[-1], torch.float32)


if __name__ == "__main__":
    unittest.main()
