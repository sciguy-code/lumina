import pytest
import numpy as np
from lumina.core.image import Image
from lumina.backends import get_backend
from lumina.backends.numpy_backend import NumpyBackend
from lumina.backends.python import PythonBackend


class TestGetBackend:
    # test the backend registry

    def test_get_numpy(self) -> None:
        backend = get_backend("numpy")
        assert isinstance(backend, NumpyBackend)

    def test_get_python(self) -> None:
        backend = get_backend("python")
        assert isinstance(backend, PythonBackend)

    def test_default_is_numpy(self) -> None:
        backend = get_backend()
        assert isinstance(backend, NumpyBackend)

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError):
            get_backend("cuda_lol")


class TestBackendsProduceSameResults:
    # the numpy and python backends should give the same results (or very close)
    # we use small images here because the python backend is slow

    @pytest.fixture
    def numpy_be(self) -> NumpyBackend:
        return NumpyBackend()

    @pytest.fixture
    def python_be(self) -> PythonBackend:
        return PythonBackend()

    @pytest.fixture
    def tiny_rgb(self) -> Image:
        # small image so the python backend doesn't take forever
        rng = np.random.RandomState(42)
        return Image(rng.randint(0, 256, (6, 6, 3), dtype=np.uint8))

    @pytest.fixture
    def tiny_gray(self) -> Image:
        rng = np.random.RandomState(42)
        return Image(rng.randint(0, 256, (6, 6), dtype=np.uint8))

    def test_to_grayscale(self, numpy_be: NumpyBackend, python_be: PythonBackend, tiny_rgb: Image) -> None:
        np_result = numpy_be.to_grayscale(tiny_rgb)
        py_result = python_be.to_grayscale(tiny_rgb)
        # allow +-1 difference due to float rounding
        diff = np.abs(np_result.data.astype(int) - py_result.data.astype(int))
        assert diff.max() <= 1

    def test_invert(self, numpy_be: NumpyBackend, python_be: PythonBackend, tiny_rgb: Image) -> None:
        np_result = numpy_be.invert(tiny_rgb)
        py_result = python_be.invert(tiny_rgb)
        np.testing.assert_array_equal(np_result.data, py_result.data)

    def test_brightness(self, numpy_be: NumpyBackend, python_be: PythonBackend, tiny_rgb: Image) -> None:
        np_result = numpy_be.adjust_brightness(tiny_rgb, 1.5)
        py_result = python_be.adjust_brightness(tiny_rgb, 1.5)
        diff = np.abs(np_result.data.astype(int) - py_result.data.astype(int))
        assert diff.max() <= 1

    def test_max_pool(self, numpy_be: NumpyBackend, python_be: PythonBackend, tiny_rgb: Image) -> None:
        np_result = numpy_be.max_pool(tiny_rgb)
        py_result = python_be.max_pool(tiny_rgb)
        np.testing.assert_array_equal(np_result.data, py_result.data)

    def test_apply_kernel(self, numpy_be: NumpyBackend, python_be: PythonBackend, tiny_gray: Image) -> None:
        # test with a simple kernel on a small image
        kernel = np.ones((3, 3), dtype=np.float32) / 9
        np_result = numpy_be.apply_kernel(tiny_gray, kernel)
        py_result = python_be.apply_kernel(tiny_gray, kernel)
        diff = np.abs(np_result.data.astype(int) - py_result.data.astype(int))
        # allow some rounding differences
        assert diff.max() <= 1
