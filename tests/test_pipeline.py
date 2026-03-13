import pytest
import os
import numpy as np
from PIL import Image as PILImage
from lumina.core.image import Image
from lumina.io.loader import load_image
from lumina.io.saver import save_image
from lumina.ops.transform import to_grayscale
from lumina.filters.convolution import gaussian_blur
from lumina.filters.edges import sobel_filter
from lumina.ops.pooling import max_pool_2x2
from lumina.filters.basic import adjust_brightness, adjust_contrast, sharpen, invert


class TestFullPipeline:
    # integration test: run the full pipeline just like the CLI does

    def test_grayscale_blur_edges_pool(self, rgb_image: Image) -> None:
        # walk through the full pipeline and make sure nothing crashes
        img = to_grayscale(rgb_image)
        img = gaussian_blur(img)
        img = sobel_filter(img)
        img = max_pool_2x2(img)

        # should end up with a small grayscale image
        assert img.channels == 1
        assert img.height == rgb_image.height // 2
        assert img.width == rgb_image.width // 2
        assert img.data.dtype == np.uint8

    def test_blur_only(self, rgb_image: Image) -> None:
        # just blur, no grayscale or edges
        result = gaussian_blur(rgb_image, size=5, sigma=1.5)
        assert result.data.shape == rgb_image.data.shape

    def test_all_basic_filters(self, rgb_image: Image) -> None:
        # stack all basic filters on top of each other
        img = adjust_brightness(rgb_image, 1.2)
        img = adjust_contrast(img, 1.3)
        img = sharpen(img)
        img = invert(img)
        # should still be a valid image with the same shape
        assert img.data.shape == rgb_image.data.shape
        assert img.data.dtype == np.uint8

    def test_save_load_roundtrip_pipeline(self, tmp_path: object) -> None:
        # create an image, run it through the pipeline, save, load, verify
        data = np.random.RandomState(42).randint(0, 256, (20, 20, 3), dtype=np.uint8)
        img = Image(data)

        # run pipeline
        img = to_grayscale(img)
        img = gaussian_blur(img)

        # save and reload
        path = os.path.join(str(tmp_path), "pipeline_output.png")
        save_image(img, path)
        loaded = load_image(path)

        # loaded image will be rgb (loader converts to rgb)
        # but all channels should have the same values since it was grayscale
        assert loaded.channels == 3
        assert os.path.exists(path)
