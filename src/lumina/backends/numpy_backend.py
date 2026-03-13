import logging
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from lumina.core.image import Image
from lumina.backends.base import Backend

# set up a logger for this module
logger = logging.getLogger(__name__)


class NumpyBackend(Backend):
    # this is the fast backend that uses numpy vectorization
    # all the heavy lifting is done with array operations, no python loops

    def apply_kernel(self, image: Image, kernel: np.ndarray) -> Image:
        # vectorized convolution using sliding windows
        k_h, k_w = kernel.shape
        data = image.data
        is_grayscale = (data.ndim == 2)

        if is_grayscale:
            data = data[:, :, None]

        h, w, c = data.shape
        pad_h = k_h // 2
        pad_w = k_w // 2

        # edge padding so borders don't go dark
        padded = np.pad(data, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='edge')

        # sliding window view - this is the magic that makes it fast
        windows = sliding_window_view(padded, (k_h, k_w), axis=(0, 1))  # type: ignore[call-overload]

        # element-wise multiply and sum across kernel dimensions
        output = np.sum(windows * kernel, axis=(3, 4))

        if is_grayscale:
            output = output[:, :, 0]

        output = np.clip(output, 0, 255).astype(np.uint8)
        return Image(output)

    def max_pool(self, image: Image) -> Image:
        # 2x2 max pooling using reshape trick (no loops needed)
        data = image.data
        is_grayscale = (data.ndim == 2)

        if is_grayscale:
            data = data[:, :, None]

        h, w, c = data.shape
        new_h = h - (h % 2)
        new_w = w - (w % 2)
        trimmed = data[:new_h, :new_w, :]

        # reshape into 2x2 blocks and take max
        reshaped = trimmed.reshape(new_h // 2, 2, new_w // 2, 2, c)
        pooled = reshaped.max(axis=(1, 3))

        if is_grayscale:
            pooled = pooled[:, :, 0]

        return Image(pooled.astype(np.uint8))

    def to_grayscale(self, image: Image) -> Image:
        if image.channels == 1:
            return image

        # bt.601 luma weights
        weights = np.array([0.299, 0.587, 0.114])
        gray = np.dot(image.data[..., :3], weights).astype(np.uint8)
        return Image(gray)

    def adjust_brightness(self, image: Image, factor: float) -> Image:
        # simple scalar multiply
        data = np.clip(image.data.astype(np.float32) * factor, 0, 255)
        return Image(data.astype(np.uint8))

    def adjust_contrast(self, image: Image, factor: float) -> Image:
        data = image.data.astype(np.float32)
        mean_val = data.mean()
        data = np.clip((data - mean_val) * factor + mean_val, 0, 255)
        return Image(data.astype(np.uint8))

    def sharpen(self, image: Image) -> Image:
        kernel = np.array([
            [ 0, -1,  0],
            [-1,  5, -1],
            [ 0, -1,  0]
        ], dtype=np.float32)
        return self.apply_kernel(image, kernel)

    def invert(self, image: Image) -> Image:
        return Image(255 - image.data)
