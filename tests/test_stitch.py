# Anima Reference for Forge Neo
# The reference picture read from ImageStitch Integrated's arguments.

import unittest

from PIL import Image

from lib_anima_ref import stitch


class FakeScript:
    def __init__(self, title, args_from, args_to):
        self._title, self.args_from, self.args_to = title, args_from, args_to

    def title(self):
        return self._title


class StitchTest(unittest.TestCase):
    def setUp(self):
        self.red, self.blue = Image.new("RGB", (8, 8), "red"), Image.new("RGB", (8, 8), "blue")
        self.scripts = [FakeScript("Anima Reference", 0, 2), FakeScript(stitch.TITLE, 2, 5)]

    def test_first_picture_of_an_enabled_gallery(self):
        args = ["x", "y", True, [(self.red, None), (self.blue, "caption")], 1024]
        image, count, problem = stitch.reference(self.scripts, args)
        self.assertIs(image, self.red)
        self.assertEqual((count, problem), (2, None))

    def test_api_gallery_is_decoded(self):
        args = ["x", "y", True, ["b64-red"], 1024]
        image, count, _ = stitch.reference(self.scripts, args, decode=lambda s: self.red if s == "b64-red" else None)
        self.assertIs(image, self.red)
        self.assertEqual(count, 1)

    def test_disabled_empty_or_missing_say_why(self):
        self.assertIn("turn on", stitch.reference(self.scripts, ["x", "y", False, [(self.red, None)], 1024])[2])
        self.assertIn("no picture", stitch.reference(self.scripts, ["x", "y", True, [], 1024])[2])
        self.assertIn("no picture", stitch.reference(self.scripts, ["x", "y", True, None, 1024])[2])
        self.assertIn("not available", stitch.reference(self.scripts[:1], ["x", "y"])[2])


if __name__ == "__main__":
    unittest.main()
