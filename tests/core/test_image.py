import pytest
import numpy as np
from lumina.core.image import Image


class TestImageCreation:
    # test that we can create images from valid numpy arrays

    def test_create_rgb_image(self) -> None:
        # should work fine with a normal rgb array
        data = np.zeros((10, 10, 3), dtype=np.uint8)
        img = Image(data)
        assert img.data.shape == (10, 10, 3)

    def test_create_grayscale_image(self) -> None:
        # should work fine with a 2d grayscale array
        data = np.zeros((10, 10), dtype=np.uint8)
        img = Image(data)
        assert img.data.shape == (10, 10)

    def test_rejects_non_array(self) -> None:
        # should blow up if you pass in a list or something
        with pytest.raises(TypeError):
            Image([1, 2, 3])

    def test_rejects_string(self) -> None:
        # definitely shouldn't accept a string
        with pytest.raises(TypeError):
            Image("not an image")

    def test_casts_to_uint8(self) -> None:
        # even if you pass in float data it should get cast to uint8
        data = np.array([[128.7, 200.3], [50.1, 0.9]])
        img = Image(data)
        assert img.data.dtype == np.uint8


class TestImageProperties:
    # test the height, width, channels properties

    def test_height_rgb(self, rgb_image: Image) -> None:
        assert rgb_image.height == 10

    def test_width_rgb(self, rgb_image: Image) -> None:
        assert rgb_image.width == 10

    def test_channels_rgb(self, rgb_image: Image) -> None:
        # rgb should have 3 channels
        assert rgb_image.channels == 3

    def test_channels_grayscale(self, grayscale_image: Image) -> None:
        # grayscale should report 1 channel
        assert grayscale_image.channels == 1

    def test_repr(self, rgb_image: Image) -> None:
        # just make sure repr doesn't crash and looks reasonable
        r = repr(rgb_image)
        assert "10x10" in r
        assert "3 ch" in r

    def test_repr_grayscale(self, grayscale_image: Image) -> None:
        r = repr(grayscale_image)
        assert "1 ch" in r

    def test_non_square_image(self) -> None:
        # make sure it handles non-square images correctly
        data = np.zeros((5, 20, 3), dtype=np.uint8)
        img = Image(data)
        assert img.height == 5
        assert img.width == 20
