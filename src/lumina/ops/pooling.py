import logging
import numpy as np
from lumina.core.image import Image

# set up a logger for this module
logger = logging.getLogger(__name__)

def max_pool_2x2(image: Image) -> Image:
    # shrink the image by half by taking the max value in each 2x2 block
    # this is a common operation in CNNs but also just a nice way to downscale
    logger.info("applying 2x2 max pooling")

    data = image.data
    is_grayscale = (data.ndim == 2)

    if is_grayscale:
        # promote to 3d so we can use the same logic for both cases
        data = data[:, :, None]

    h, w, c = data.shape

    # trim edges so height and width are evenly divisible by 2
    # e.g. width 1001 becomes 1000, we just drop the last column
    new_h = h - (h % 2)
    new_w = w - (w % 2)
    trimmed_data = data[:new_h, :new_w, :]

    # reshape into non-overlapping 2x2 blocks
    # (h, w, c) -> (h/2, 2, w/2, 2, c)
    reshaped = trimmed_data.reshape(new_h // 2, 2, new_w // 2, 2, c)

    # take the max over the 2x2 block dimensions (axis 1 and 3)
    pooled_data = reshaped.max(axis=(1, 3))

    if is_grayscale:
        # squeeze back to 2d
        pooled_data = pooled_data[:, :, 0]

    logger.info(f"pooled from {w}x{h} to {pooled_data.shape[1]}x{pooled_data.shape[0]}")
    return Image(pooled_data.astype(np.uint8))
