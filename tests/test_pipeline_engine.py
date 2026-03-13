import pytest
import numpy as np
from lumina.core.image import Image
from lumina.pipeline.engine import Pipeline
from lumina.ops.transform import to_grayscale
from lumina.filters.convolution import gaussian_blur
from lumina.filters.basic import invert


class TestPipelineEngine:
    # test the pipeline engine itself

    def test_empty_pipeline(self, rgb_image: Image) -> None:
        # running an empty pipeline should return the image unchanged
        pipe = Pipeline()
        result = pipe.run(rgb_image)
        np.testing.assert_array_equal(result.data, rgb_image.data)

    def test_single_step(self, rgb_image: Image) -> None:
        # a single step pipeline should work like calling the function directly
        pipe = Pipeline().add(to_grayscale)
        result = pipe.run(rgb_image)
        expected = to_grayscale(rgb_image)
        np.testing.assert_array_equal(result.data, expected.data)

    def test_multi_step(self, rgb_image: Image) -> None:
        # chain multiple steps together
        pipe = Pipeline().add(to_grayscale).add(gaussian_blur).add(invert)
        result = pipe.run(rgb_image)
        # should be grayscale, blurred, and inverted
        assert result.channels == 1
        assert result.data.dtype == np.uint8

    def test_step_with_kwargs(self, rgb_image: Image) -> None:
        # should pass kwargs through to the step function
        pipe = Pipeline().add(gaussian_blur, size=5, sigma=2.0)
        result = pipe.run(rgb_image)
        assert result.data.shape == rgb_image.data.shape

    def test_len(self) -> None:
        pipe = Pipeline().add(to_grayscale).add(gaussian_blur)
        assert len(pipe) == 2

    def test_repr(self) -> None:
        pipe = Pipeline().add(to_grayscale).add(invert)
        r = repr(pipe)
        assert "to_grayscale" in r
        assert "invert" in r
