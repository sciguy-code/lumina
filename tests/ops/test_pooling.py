import pytest
import numpy as np
from lumina.core.image import Image
from lumina.ops.pooling import max_pool_2x2


class TestMaxPool2x2:
    # test the 2x2 max pooling

    def test_halves_dimensions_rgb(self, rgb_image: Image) -> None:
        # output should be half the size in each dimension
        result = max_pool_2x2(rgb_image)
        assert result.height == rgb_image.height // 2
        assert result.width == rgb_image.width // 2

    def test_halves_dimensions_grayscale(self, grayscale_image: Image) -> None:
        result = max_pool_2x2(grayscale_image)
        assert result.height == grayscale_image.height // 2
        assert result.width == grayscale_image.width // 2

    def test_preserves_channels(self, rgb_image: Image) -> None:
        # should keep the same number of channels
        result = max_pool_2x2(rgb_image)
        assert result.channels == rgb_image.channels

    def test_output_is_uint8(self, rgb_image: Image) -> None:
        result = max_pool_2x2(rgb_image)
        assert result.data.dtype == np.uint8

    def test_correct_max_values(self) -> None:
        # test with known values to make sure it picks the max
        data = np.array([
            [10, 20, 30, 40],
            [50, 60, 70, 80],
            [90, 100, 110, 120],
            [130, 140, 150, 160],
        ], dtype=np.uint8)
        img = Image(data)
        result = max_pool_2x2(img)
        # top-left 2x2 block: max(10,20,50,60) = 60
        assert result.data[0, 0] == 60
        # top-right 2x2 block: max(30,40,70,80) = 80
        assert result.data[0, 1] == 80
        # bottom-left: max(90,100,130,140) = 140
        assert result.data[1, 0] == 140
        # bottom-right: max(110,120,150,160) = 160
        assert result.data[1, 1] == 160

    def test_odd_dimensions_trimmed(self) -> None:
        # odd dimensions should be trimmed (e.g. 5x5 -> 4x4 -> pooled to 2x2)
        data = np.zeros((5, 5), dtype=np.uint8)
        img = Image(data)
        result = max_pool_2x2(img)
        assert result.height == 2
        assert result.width == 2

    def test_1x1_image(self) -> None:
        # edge case: 1x1 image gets trimmed to 0x0 which should handle gracefully
        data = np.array([[42]], dtype=np.uint8)
        img = Image(data)
        result = max_pool_2x2(img)
        # trimmed to 0x0 so result should be empty
        assert result.height == 0
