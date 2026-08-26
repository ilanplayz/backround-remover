import unittest
from pathlib import Path

import numpy as np
from PIL import Image

import app


class BackgroundRemoverTests(unittest.TestCase):
    def setUp(self):
        self.image = np.full((180, 260, 3), 245, dtype=np.uint8)
        self.image[35:155, 75:190] = (35, 90, 205)

    def test_mask_matches_original_size(self):
        mask = app.create_foreground_mask(self.image)
        self.assertEqual(mask.shape, self.image.shape[:2])
        self.assertEqual(mask.dtype, np.uint8)
        self.assertGreater(int(mask[90, 130]), int(mask[5, 5]))

    def test_upload_creates_downloadable_rgba_png(self):
        rgba, path, status = app.remove_background(self.image)
        self.assertEqual(rgba.shape, (180, 260, 4))
        self.assertIn("Background removed", status)
        with Image.open(path) as exported:
            self.assertEqual(exported.mode, "RGBA")
            self.assertEqual(exported.size, (260, 180))
        Path(path).unlink(missing_ok=True)

    def test_grayscale_input_is_supported(self):
        grayscale = np.full((120, 160), 220, dtype=np.uint8)
        grayscale[25:100, 50:115] = 40
        normalized = app.normalize_image(grayscale)
        self.assertEqual(normalized.shape, (120, 160, 3))

    def test_float_input_is_supported(self):
        floating = np.ones((80, 90, 3), dtype=np.float32)
        floating[20:65, 30:70] = (0.1, 0.3, 0.8)
        normalized = app.normalize_image(floating)
        self.assertEqual(normalized.dtype, np.uint8)
        self.assertEqual(normalized.shape, (80, 90, 3))

    def test_rgba_input_preserves_existing_transparency(self):
        rgba_input = np.dstack(
            [
                self.image,
                np.full(self.image.shape[:2], 255, dtype=np.uint8),
            ]
        )
        rgba_input[:20, :20, 3] = 0
        rgba, path, _ = app.remove_background(rgba_input)
        self.assertEqual(rgba.shape, (180, 260, 4))
        self.assertEqual(int(rgba[5, 5, 3]), 0)
        Path(path).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
