import pytest
import numpy as np
from lumina.core.image import Image
from lumina.filters.edges import sobel_filter


class TestSobelFilter:
    # test the sobel edge detection

    def test_uniform_image_no_edges(self) -> None:
        # a completely uniform image should have no edges at all
        data = np.full((10, 10), 128, dtype=np.uint8)
        img = Image(data)
        result = sobel_filter(img)
        # all pixels should be 0 since there's no gradient
        assert result.data.max() == 0

    def test_output_shape_preserved(self, grayscale_image: Image) -> None:
        result = sobel_filter(grayscale_image)
        assert result.data.shape == grayscale_image.data.shape

    def test_output_is_uint8(self, grayscale_image: Image) -> None:
        result = sobel_filter(grayscale_image)
        assert result.data.dtype == np.uint8

    def test_vertical_edge_detected(self) -> None:
        # create an image with a sharp vertical edge down the middle
        data = np.zeros((10, 10), dtype=np.uint8)
        data[:, 5:] = 255
        img = Image(data)
        result = sobel_filter(img)
        # the edge columns should have high values
        # the middle area where the edge is should be bright
        assert result.data.max() > 100

    def test_horizontal_edge_detected(self) -> None:
        # create an image with a sharp horizontal edge
        data = np.zeros((10, 10), dtype=np.uint8)
        data[5:, :] = 255
        img = Image(data)
        result = sobel_filter(img)
        # should detect the horizontal edge
        assert result.data.max() > 100

    def test_output_range(self, grayscale_image: Image) -> None:
        # output should be normalized to 0-255
        result = sobel_filter(grayscale_image)
        assert result.data.min() >= 0
        assert result.data.max() <= 255
