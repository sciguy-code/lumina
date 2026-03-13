import pytest
import numpy as np
from lumina.core.image import Image
from lumina.filters.convolution import apply_kernel, gaussian_blur, build_gaussian_kernel


class TestApplyKernel:
    # test the core convolution function

    def test_identity_kernel_rgb(self, rgb_image: Image) -> None:
        # an identity kernel should give back roughly the same image
        kernel = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32)
        result = apply_kernel(rgb_image, kernel)
        # should be identical since we're just copying center pixel
        np.testing.assert_array_equal(result.data, rgb_image.data)

    def test_identity_kernel_grayscale(self, grayscale_image: Image) -> None:
        # same thing but for grayscale
        kernel = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32)
        result = apply_kernel(grayscale_image, kernel)
        np.testing.assert_array_equal(result.data, grayscale_image.data)

    def test_output_shape_preserved_rgb(self, rgb_image: Image) -> None:
        # output should have the same shape as input
        kernel = np.ones((3, 3), dtype=np.float32) / 9
        result = apply_kernel(rgb_image, kernel)
        assert result.data.shape == rgb_image.data.shape

    def test_output_shape_preserved_grayscale(self, grayscale_image: Image) -> None:
        kernel = np.ones((3, 3), dtype=np.float32) / 9
        result = apply_kernel(grayscale_image, kernel)
        assert result.data.shape == grayscale_image.data.shape

    def test_output_is_uint8(self, rgb_image: Image) -> None:
        # output should always be uint8 no matter what
        kernel = np.ones((3, 3), dtype=np.float32) / 9
        result = apply_kernel(rgb_image, kernel)
        assert result.data.dtype == np.uint8

    def test_5x5_kernel(self, rgb_image: Image) -> None:
        # should also work with larger kernels
        kernel = np.ones((5, 5), dtype=np.float32) / 25
        result = apply_kernel(rgb_image, kernel)
        assert result.data.shape == rgb_image.data.shape


class TestBuildGaussianKernel:
    # test the kernel builder function

    def test_3x3_kernel_shape(self) -> None:
        kernel = build_gaussian_kernel(3, 1.0)
        assert kernel.shape == (3, 3)

    def test_5x5_kernel_shape(self) -> None:
        kernel = build_gaussian_kernel(5, 1.0)
        assert kernel.shape == (5, 5)

    def test_kernel_sums_to_one(self) -> None:
        # a gaussian kernel should sum to 1 (preserves brightness)
        kernel = build_gaussian_kernel(5, 1.5)
        assert abs(kernel.sum() - 1.0) < 1e-10

    def test_center_is_max(self) -> None:
        # the center of the kernel should be the largest value
        kernel = build_gaussian_kernel(5, 1.0)
        assert kernel[2, 2] == kernel.max()

    def test_kernel_is_symmetric(self) -> None:
        # gaussian should be symmetric in all directions
        kernel = build_gaussian_kernel(5, 1.0)
        np.testing.assert_array_almost_equal(kernel, kernel.T)

    def test_even_size_raises(self) -> None:
        # even kernel sizes don't make sense, should raise an error
        with pytest.raises(ValueError):
            build_gaussian_kernel(4, 1.0)


class TestGaussianBlur:
    # test the blur function end-to-end

    def test_blur_preserves_shape_rgb(self, rgb_image: Image) -> None:
        result = gaussian_blur(rgb_image)
        assert result.data.shape == rgb_image.data.shape

    def test_blur_preserves_shape_grayscale(self, grayscale_image: Image) -> None:
        result = gaussian_blur(grayscale_image)
        assert result.data.shape == grayscale_image.data.shape

    def test_blur_output_is_uint8(self, rgb_image: Image) -> None:
        result = gaussian_blur(rgb_image)
        assert result.data.dtype == np.uint8

    def test_blur_with_custom_size(self, rgb_image: Image) -> None:
        # should work with size=5 too
        result = gaussian_blur(rgb_image, size=5, sigma=1.5)
        assert result.data.shape == rgb_image.data.shape

    def test_blur_reduces_noise(self) -> None:
        # blurring a noisy image should reduce the variance
        rng = np.random.RandomState(42)
        noisy = Image(rng.randint(0, 256, (20, 20), dtype=np.uint8))
        blurred = gaussian_blur(noisy)
        # variance of blurred image should be lower
        assert blurred.data.astype(float).var() < noisy.data.astype(float).var()
