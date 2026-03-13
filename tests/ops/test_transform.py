import pytest
import numpy as np
from lumina.core.image import Image
from lumina.ops.transform import to_grayscale


class TestToGrayscale:
    # test the grayscale conversion

    def test_rgb_to_grayscale(self, rgb_image: Image) -> None:
        # should go from 3 channels down to 1
        result = to_grayscale(rgb_image)
        assert result.channels == 1
        assert result.data.ndim == 2

    def test_output_shape(self, rgb_image: Image) -> None:
        # height and width should stay the same
        result = to_grayscale(rgb_image)
        assert result.height == rgb_image.height
        assert result.width == rgb_image.width

    def test_output_is_uint8(self, rgb_image: Image) -> None:
        result = to_grayscale(rgb_image)
        assert result.data.dtype == np.uint8

    def test_already_grayscale_returns_same(self, grayscale_image: Image) -> None:
        # if it's already grayscale, should just return it untouched
        result = to_grayscale(grayscale_image)
        assert result is grayscale_image

    def test_known_values(self) -> None:
        # test with a known rgb value to verify the luma weights are correct
        # pure red (255, 0, 0) should give: 0.299 * 255 = 76.245 -> 76
        data = np.array([[[255, 0, 0]]], dtype=np.uint8)
        img = Image(data)
        result = to_grayscale(img)
        # should be around 76 (0.299 * 255)
        assert abs(int(result.data[0, 0]) - 76) <= 1

    def test_pure_white(self) -> None:
        # pure white should stay close to 255
        data = np.array([[[255, 255, 255]]], dtype=np.uint8)
        img = Image(data)
        result = to_grayscale(img)
        # 0.299*255 + 0.587*255 + 0.114*255 = 255
        assert result.data[0, 0] == 255

    def test_pure_black(self) -> None:
        # pure black should stay at 0
        data = np.array([[[0, 0, 0]]], dtype=np.uint8)
        img = Image(data)
        result = to_grayscale(img)
        assert result.data[0, 0] == 0
