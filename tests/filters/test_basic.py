import pytest
import numpy as np
from lumina.core.image import Image
from lumina.filters.basic import adjust_brightness, adjust_contrast, sharpen, invert


class TestAdjustBrightness:
    # test the brightness adjustment

    def test_brighter(self, rgb_image: Image) -> None:
        # factor > 1 should make it brighter (higher pixel values on average)
        result = adjust_brightness(rgb_image, 1.5)
        # at minimum, the mean should go up (or stay if already saturated)
        assert result.data.astype(float).mean() >= rgb_image.data.astype(float).mean()

    def test_darker(self, rgb_image: Image) -> None:
        # factor < 1 should make it darker
        result = adjust_brightness(rgb_image, 0.5)
        assert result.data.astype(float).mean() <= rgb_image.data.astype(float).mean()

    def test_no_change(self, rgb_image: Image) -> None:
        # factor = 1 should leave it unchanged
        result = adjust_brightness(rgb_image, 1.0)
        np.testing.assert_array_equal(result.data, rgb_image.data)

    def test_output_clamped(self) -> None:
        # even with a huge factor, values should stay in 0-255
        data = np.full((4, 4, 3), 200, dtype=np.uint8)
        img = Image(data)
        result = adjust_brightness(img, 5.0)
        assert result.data.max() <= 255

    def test_output_is_uint8(self, rgb_image: Image) -> None:
        result = adjust_brightness(rgb_image, 1.5)
        assert result.data.dtype == np.uint8


class TestAdjustContrast:
    # test the contrast adjustment

    def test_more_contrast(self) -> None:
        # higher factor should increase the spread of values
        data = np.array([[100, 150], [100, 150]], dtype=np.uint8)
        img = Image(data)
        result = adjust_contrast(img, 2.0)
        # the range should be wider (or at least not smaller)
        orig_range = img.data.astype(float).max() - img.data.astype(float).min()
        result_range = result.data.astype(float).max() - result.data.astype(float).min()
        assert result_range >= orig_range

    def test_no_change(self, rgb_image: Image) -> None:
        # factor = 1 should be roughly the same (might have tiny rounding diffs)
        result = adjust_contrast(rgb_image, 1.0)
        # allow for small rounding differences
        diff = np.abs(result.data.astype(float) - rgb_image.data.astype(float))
        assert diff.max() <= 1

    def test_output_is_uint8(self, rgb_image: Image) -> None:
        result = adjust_contrast(rgb_image, 1.5)
        assert result.data.dtype == np.uint8


class TestSharpen:
    # test the sharpening filter

    def test_output_shape(self, rgb_image: Image) -> None:
        result = sharpen(rgb_image)
        assert result.data.shape == rgb_image.data.shape

    def test_output_is_uint8(self, rgb_image: Image) -> None:
        result = sharpen(rgb_image)
        assert result.data.dtype == np.uint8

    def test_sharpen_grayscale(self, grayscale_image: Image) -> None:
        result = sharpen(grayscale_image)
        assert result.data.shape == grayscale_image.data.shape


class TestInvert:
    # test color inversion

    def test_invert_values(self) -> None:
        # inverting should flip values: 0 -> 255, 255 -> 0
        data = np.array([[0, 128, 255]], dtype=np.uint8)
        img = Image(data)
        result = invert(img)
        expected = np.array([[255, 127, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(result.data, expected)

    def test_double_invert(self, rgb_image: Image) -> None:
        # inverting twice should give back the original
        result = invert(invert(rgb_image))
        np.testing.assert_array_equal(result.data, rgb_image.data)

    def test_output_shape(self, rgb_image: Image) -> None:
        result = invert(rgb_image)
        assert result.data.shape == rgb_image.data.shape

    def test_output_is_uint8(self, rgb_image: Image) -> None:
        result = invert(rgb_image)
        assert result.data.dtype == np.uint8
