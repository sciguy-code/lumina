import logging
import numpy as np
from lumina.core.image import Image

# set up a logger for this module
logger = logging.getLogger(__name__)

def to_grayscale(image: Image) -> Image:
    # luma formula: gray = 0.299r + 0.587g + 0.114b
    # these weights come from how human eyes perceive brightness

    if image.channels == 1:
        # already grayscale, just hand it back
        logger.info("image is already grayscale, skipping conversion")
        return image

    # the standard bt.601 luma coefficients
    weights = np.array([0.299, 0.587, 0.114])

    # dot product across the color channels to get a single brightness value
    grayscale_data = np.dot(image.data[..., :3], weights)

    # make sure we're back to uint8 so it plays nice with everything else
    grayscale_data = grayscale_data.astype(np.uint8)

    logger.info("converted image to grayscale")
    return Image(grayscale_data)
