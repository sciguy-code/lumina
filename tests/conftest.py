import pytest
import numpy as np
from PIL import Image as PILImage
import tempfile
import os
from lumina.core.image import Image


# --- shared fixtures for the whole test suite ---

@pytest.fixture
def rgb_image() -> Image:
    # a small 10x10 rgb image with random pixel values
    # using a seed so tests are reproducible
    rng = np.random.RandomState(42)
    data = rng.randint(0, 256, (10, 10, 3), dtype=np.uint8)
    return Image(data)


@pytest.fixture
def grayscale_image() -> Image:
    # a small 10x10 grayscale image
    rng = np.random.RandomState(42)
    data = rng.randint(0, 256, (10, 10), dtype=np.uint8)
    return Image(data)


@pytest.fixture
def small_rgb_image() -> Image:
    # a tiny 4x4 rgb image for testing where we need to inspect exact values
    data = np.zeros((4, 4, 3), dtype=np.uint8)
    # make a simple pattern: top half bright, bottom half dark
    data[:2, :, :] = 200
    data[2:, :, :] = 50
    return Image(data)


@pytest.fixture
def small_grayscale_image() -> Image:
    # a tiny 4x4 grayscale image
    data = np.array([
        [100, 150, 100, 150],
        [150, 200, 150, 200],
        [100, 150, 100, 150],
        [150, 200, 150, 200],
    ], dtype=np.uint8)
    return Image(data)


@pytest.fixture
def tmp_image_path(tmp_path: object) -> str:
    # creates a temporary png file we can use for io testing
    path = os.path.join(str(tmp_path), "test_img.png")
    # make a small image and save it
    pil_img = PILImage.fromarray(np.zeros((10, 10, 3), dtype=np.uint8))
    pil_img.save(path)
    return path
