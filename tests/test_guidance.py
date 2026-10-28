# Anima Reference for Forge Neo
# Text and reference guidance, APG and rescale, plus the letterbox.

import unittest

import torch
from PIL import Image

from lib_anima_ref import guidance
from lib_anima_ref.letterbox import letterbox, to_pixels


class GuidanceTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.pos, self.neg, self.base = (torch.randn(2, 4, 1, 6, 6) for _ in range(3))

    def test_ref_guidance_one_is_plain_cfg(self):
        out = guidance.combine(self.pos, self.neg, None, 4.5, 1.0, "none")
        torch.testing.assert_close(out, self.neg + 4.5 * (self.pos - self.neg))
        self.assertFalse(guidance.needs_base(1.0))
        self.assertTrue(guidance.needs_base(4.5))

    def test_separate_guidance(self):
        out = guidance.combine(self.pos, self.neg, self.base, 3.0, 5.0, "none")
        torch.testing.assert_close(out, self.base + 5.0 * (self.neg - self.base) + 3.0 * (self.pos - self.neg))

    def test_apg_drops_the_update_parallel_to_pos(self):
        out = guidance.combine(self.pos, self.neg, self.base, 4.5, 4.5, "apg")
        update = (out - self.pos).double().flatten(1)
        dots = (update * self.pos.double().flatten(1)).sum(1)
        torch.testing.assert_close(dots, torch.zeros(2, dtype=torch.float64), atol=1e-4, rtol=0)

    def test_rescale_phi_one_matches_pos_std(self):
        out = guidance.combine(self.pos, self.neg, self.base, 4.5, 4.5, "rescale", 1.0)
        torch.testing.assert_close(out.std(dim=(1, 2, 3, 4)), self.pos.std(dim=(1, 2, 3, 4)))


class LetterboxTest(unittest.TestCase):
    def test_fits_and_centers_on_the_fill(self):
        src = Image.new("RGB", (100, 50), (10, 20, 30))
        canvas = letterbox(src, 64, 96, 255)
        self.assertEqual(canvas.size, (64, 96))
        self.assertEqual(canvas.getpixel((0, 0)), (255, 255, 255))
        self.assertEqual(canvas.getpixel((32, 48)), (10, 20, 30))
        pixels = to_pixels(canvas)
        self.assertEqual(pixels.shape, (1, 96, 64, 3))
        self.assertAlmostEqual(pixels.max().item(), 1.0)

    def test_grey_fill_and_rgba_source(self):
        canvas = letterbox(Image.new("RGBA", (10, 40), (0, 0, 0, 0)), 64, 64, 128)
        self.assertEqual(canvas.getpixel((0, 32)), (128, 128, 128))


if __name__ == "__main__":
    unittest.main()
