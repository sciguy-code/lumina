import logging
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from lumina.core.image import Image

# set up a logger for this module
logger = logging.getLogger(__name__)

def apply_kernel(image: Image, kernel: np.ndarray) -> Image:
    # this is the core convolution operation
    # it slides a kernel over every pixel and sums up the weighted neighbors

    # grab kernel dimensions
    k_h, k_w = kernel.shape
    data = image.data
    is_grayscale = (data.ndim == 2)

    if is_grayscale:
        # promote to (h, w, 1) so the same code path works for both
        data = data[:, :, None]

    h, w, c = data.shape

    # figure out how much padding we need so the kernel can reach border pixels
    pad_h = k_h // 2
    pad_w = k_w // 2

    # mode='edge' repeats the border values, avoids those ugly dark borders
    padded_image = np.pad(data, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='edge')

    # sliding_window_view gives us a 5d view: (height, width, channels, kernel_h, kernel_w)
    # no copies are made here which is pretty neat
    windows = sliding_window_view(padded_image, (k_h, k_w), axis=(0, 1))  # type: ignore[call-overload]

    # multiply each window by the kernel and sum up to get the output pixel
    output_data = np.sum(windows * kernel, axis=(3, 4))

    if is_grayscale:
        # squeeze back down to (h, w) for grayscale
        output_data = output_data[:, :, 0]

    # clamp to valid pixel range and cast back to uint8
    output_data = np.clip(output_data, 0, 255).astype(np.uint8)

    return Image(output_data)


def build_gaussian_kernel(size: int = 3, sigma: float = 1.0) -> np.ndarray:
    # build a proper gaussian kernel from sigma instead of hardcoding values
    # this is the 2d gaussian formula: G(x,y) = exp(-(x^2 + y^2) / (2*sigma^2))

    if size % 2 == 0:
        raise ValueError(f"kernel size must be odd, got {size}")

    # create a grid of (x, y) coordinates centered at zero
    half = size // 2
    ax = np.arange(-half, half + 1, dtype=np.float64)
    xx, yy = np.meshgrid(ax, ax)

    # apply the gaussian formula
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))

    # normalize so all values sum to 1 (preserves brightness)
    kernel = kernel / kernel.sum()

    return kernel  # type: ignore[no-any-return]


def gaussian_blur(image: Image, size: int = 3, sigma: float = 1.0) -> Image:
    # apply gaussian blur with a configurable kernel size and sigma
    logger.info(f"applying {size}x{size} gaussian blur (sigma={sigma})")

    kernel = build_gaussian_kernel(size, sigma)
    return apply_kernel(image, kernel)
