from __future__ import annotations

import io
import unittest

from PIL import Image

from webapp.inference import InputError, decode_image


def image_bytes(fmt: str = "PNG", size: tuple[int, int] = (64, 48)) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", size, "white").save(output, fmt)
    return output.getvalue()


class ValidationTest(unittest.TestCase):
    def test_valid_png(self) -> None:
        image, width, height = decode_image(image_bytes(), "image/png")
        self.assertEqual((width, height), (64, 48))
        self.assertEqual(image.shape, (48, 64, 3))

    def test_fake_jpeg_is_rejected(self) -> None:
        with self.assertRaises(InputError):
            decode_image(b"not an image", "image/jpeg")

    def test_mime_is_checked(self) -> None:
        with self.assertRaises(InputError):
            decode_image(image_bytes(), "application/octet-stream")

    def test_small_image_is_rejected(self) -> None:
        with self.assertRaises(InputError):
            decode_image(image_bytes(size=(10, 10)), "image/png")

    def test_large_dimension_is_rejected(self) -> None:
        with self.assertRaises(InputError):
            decode_image(image_bytes(size=(4097, 32)), "image/png")


if __name__ == "__main__":
    unittest.main()
