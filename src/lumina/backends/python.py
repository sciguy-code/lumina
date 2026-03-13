import logging
import math
import numpy as np
from lumina.core.image import Image
from lumina.backends.base import Backend

# set up a logger for this module
logger = logging.getLogger(__name__)


class PythonBackend(Backend):
    # this is the slow-but-readable backend that uses pure python loops
    # no numpy vectorization here, just nested for loops
    # useful for understanding what the algorithms actually do step by step

    def apply_kernel(self, image: Image, kernel: np.ndarray) -> Image:
        # manually slide the kernel over every pixel using python loops
        k_h, k_w = kernel.shape
        data = image.data
        is_grayscale = (data.ndim == 2)

        if is_grayscale:
            data = data[:, :, None]

        h, w, c = data.shape
        pad_h = k_h // 2
        pad_w = k_w // 2

        # manually pad the image by repeating edge pixels
        padded = np.pad(data, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='edge')

        # allocate the output array
        output = np.zeros((h, w, c), dtype=np.float64)

        # the classic triple-nested loop - slow but easy to understand
        for y in range(h):
            for x in range(w):
                for ch in range(c):
                    total = 0.0
                    for ky in range(k_h):
                        for kx in range(k_w):
                            pixel_val = float(padded[y + ky, x + kx, ch])
                            total += pixel_val * kernel[ky, kx]
                    output[y, x, ch] = total

        if is_grayscale:
            output = output[:, :, 0]

        output = np.clip(output, 0, 255).astype(np.uint8)
        return Image(output)

    def max_pool(self, image: Image) -> Image:
        # 2x2 max pooling with explicit loops
        data = image.data
        is_grayscale = (data.ndim == 2)

        if is_grayscale:
            data = data[:, :, None]

        h, w, c = data.shape
        new_h = h - (h % 2)
        new_w = w - (w % 2)

        out_h = new_h // 2
        out_w = new_w // 2
        output = np.zeros((out_h, out_w, c), dtype=np.uint8)

        # walk through each 2x2 block and pick the max
        for y in range(out_h):
            for x in range(out_w):
                for ch in range(c):
                    block = [
                        data[y*2, x*2, ch],
                        data[y*2+1, x*2, ch],
                        data[y*2, x*2+1, ch],
                        data[y*2+1, x*2+1, ch],
                    ]
                    output[y, x, ch] = max(block)

        if is_grayscale:
            output = output[:, :, 0]

        return Image(output)

    def to_grayscale(self, image: Image) -> Image:
        if image.channels == 1:
            return image

        h, w = image.height, image.width
        output = np.zeros((h, w), dtype=np.uint8)

        # apply luma weights pixel by pixel
        for y in range(h):
            for x in range(w):
                r = float(image.data[y, x, 0])
                g = float(image.data[y, x, 1])
                b = float(image.data[y, x, 2])
                gray = 0.299 * r + 0.587 * g + 0.114 * b
                output[y, x] = min(255, max(0, int(gray)))

        return Image(output)

    def adjust_brightness(self, image: Image, factor: float) -> Image:
        data = image.data.copy()
        h, w = data.shape[0], data.shape[1]
        channels = 1 if data.ndim == 2 else data.shape[2]

        if data.ndim == 2:
            data = data[:, :, None]

        output = np.zeros_like(data)
        for y in range(h):
            for x in range(w):
                for ch in range(channels):
                    val = float(data[y, x, ch]) * factor
                    output[y, x, ch] = min(255, max(0, int(val)))

        if channels == 1:
            output = output[:, :, 0]

        return Image(output)

    def adjust_contrast(self, image: Image, factor: float) -> Image:
        # compute mean the slow way
        data = image.data
        total = 0.0
        count = 0

        if data.ndim == 2:
            for y in range(data.shape[0]):
                for x in range(data.shape[1]):
                    total += float(data[y, x])
                    count += 1
        else:
            for y in range(data.shape[0]):
                for x in range(data.shape[1]):
                    for ch in range(data.shape[2]):
                        total += float(data[y, x, ch])
                        count += 1

        mean_val = total / count

        # apply contrast scaling around the mean
        is_grayscale = (data.ndim == 2)
        if is_grayscale:
            data = data[:, :, None]

        h, w, c = data.shape
        output = np.zeros_like(data)
        for y in range(h):
            for x in range(w):
                for ch in range(c):
                    val = (float(data[y, x, ch]) - mean_val) * factor + mean_val
                    output[y, x, ch] = min(255, max(0, int(val)))

        if is_grayscale:
            output = output[:, :, 0]

        return Image(output)

    def sharpen(self, image: Image) -> Image:
        # use the same sharpening kernel, but go through our slow apply_kernel
        kernel = np.array([
            [ 0, -1,  0],
            [-1,  5, -1],
            [ 0, -1,  0]
        ], dtype=np.float32)
        return self.apply_kernel(image, kernel)

    def invert(self, image: Image) -> Image:
        # subtract each pixel from 255 one by one
        data = image.data.copy()
        if data.ndim == 2:
            for y in range(data.shape[0]):
                for x in range(data.shape[1]):
                    data[y, x] = 255 - data[y, x]
        else:
            for y in range(data.shape[0]):
                for x in range(data.shape[1]):
                    for ch in range(data.shape[2]):
                        data[y, x, ch] = 255 - data[y, x, ch]
        return Image(data)
