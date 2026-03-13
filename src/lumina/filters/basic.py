import logging
import numpy as np
from lumina.core.image import Image
from lumina.filters.convolution import apply_kernel

# set up a logger for this module
logger = logging.getLogger(__name__)

def adjust_brightness(image: Image, factor: float) -> Image:
    # multiply all pixel values by a factor
    # factor > 1 = brighter, factor < 1 = darker, factor = 1 = no change
    logger.info(f"adjusting brightness by factor {factor}")

    # work in float so we don't overflow during multiplication
    data = image.data.astype(np.float32) * factor

    # clamp back to valid pixel range
    data = np.clip(data, 0, 255).astype(np.uint8)
    return Image(data)


def adjust_contrast(image: Image, factor: float) -> Image:
    # adjust contrast by scaling pixel distance from the mean intensity
    # factor > 1 = more contrast, factor < 1 = less contrast
    logger.info(f"adjusting contrast by factor {factor}")

    data = image.data.astype(np.float32)

    # compute the mean intensity across all pixels
    mean_val = data.mean()

    # push pixels away from (or toward) the mean
    data = (data - mean_val) * factor + mean_val

    # clamp back to valid range
    data = np.clip(data, 0, 255).astype(np.uint8)
    return Image(data)


def sharpen(image: Image) -> Image:
    # apply a 3x3 sharpening kernel
    # this works by emphasizing the center pixel and subtracting neighbors
    logger.info("applying sharpening filter")

    kernel = np.array([
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0]
    ], dtype=np.float32)

    return apply_kernel(image, kernel)


def invert(image: Image) -> Image:
    # flip all the colors: white becomes black, black becomes white
    # just subtract each pixel value from 255
    logger.info("inverting image colors")

    data: np.ndarray = 255 - image.data
    return Image(data)
