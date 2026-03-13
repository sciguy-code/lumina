import pytest
import os
import numpy as np
from PIL import Image as PILImage
from lumina.core.image import Image
from lumina.io.loader import load_image
from lumina.io.saver import save_image


class TestLoadImage:
    # test the image loader

    def test_load_valid_image(self, tmp_image_path: str) -> None:
        # should load a valid image file without crashing
        img = load_image(tmp_image_path)
        assert isinstance(img, Image)

    def test_load_returns_rgb(self, tmp_image_path: str) -> None:
        # should always convert to RGB (3 channels)
        img = load_image(tmp_image_path)
        assert img.channels == 3

    def test_load_nonexistent_raises(self) -> None:
        # should raise FileNotFoundError for missing files
        with pytest.raises(FileNotFoundError):
            load_image("/nonexistent/path/to/image.png")

    def test_load_correct_dimensions(self, tmp_path: object) -> None:
        # make a specific sized image and verify dimensions
        path = os.path.join(str(tmp_path), "sized.png")
        pil_img = PILImage.fromarray(np.zeros((50, 30, 3), dtype=np.uint8))
        pil_img.save(path)
        img = load_image(path)
        assert img.height == 50
        assert img.width == 30


class TestSaveImage:
    # test the image saver

    def test_save_creates_file(self, rgb_image: Image, tmp_path: object) -> None:
        # saving should create a file on disk
        path = os.path.join(str(tmp_path), "output.png")
        save_image(rgb_image, path)
        assert os.path.exists(path)

    def test_roundtrip(self, tmp_path: object) -> None:
        # save then load should give back roughly the same image
        data = np.array([[[128, 64, 32]]], dtype=np.uint8)
        img = Image(data)
        path = os.path.join(str(tmp_path), "roundtrip.png")
        save_image(img, path)
        loaded = load_image(path)
        # should be close (PNG is lossless so should be exact)
        np.testing.assert_array_equal(loaded.data, img.data)

    def test_save_grayscale(self, grayscale_image: Image, tmp_path: object) -> None:
        # saving a grayscale image should work too
        path = os.path.join(str(tmp_path), "gray.png")
        save_image(grayscale_image, path)
        assert os.path.exists(path)
