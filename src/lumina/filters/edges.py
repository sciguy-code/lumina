import logging
import numpy as np
from lumina.core.image import Image
from lumina.filters.convolution import apply_kernel

# set up a logger for this module
logger = logging.getLogger(__name__)

def sobel_filter(image: Image) -> Image:
    # detect edges using the sobel operator
    # we compute gradients in x and y directions, then combine them
    logger.info("applying sobel edge detection")

    # kx picks up left-to-right intensity changes (vertical edges)
    Kx = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float32)

    # ky picks up top-to-bottom intensity changes (horizontal edges)
    Ky = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ], dtype=np.float32)

    # need float32 here because gradients can go negative
    img_float_data = image.data.astype(np.float32)
    img_float_wrapper = Image(img_float_data)

    # compute gradient in each direction
    gx_image = apply_kernel(img_float_wrapper, Kx)
    gx = gx_image.data.astype(np.float32)

    gy_image = apply_kernel(img_float_wrapper, Ky)
    gy = gy_image.data.astype(np.float32)

    # combine both gradients into the edge magnitude
    magnitude = np.sqrt(gx**2 + gy**2)

    # normalize so the strongest edge becomes 255
    if magnitude.max() > 0:
        magnitude = (magnitude / magnitude.max()) * 255

    logger.info("sobel edge detection complete")
    return Image(magnitude.astype(np.uint8))
