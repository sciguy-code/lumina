import logging
import numpy as np
from lumina.core.image import Image
from lumina.filters.convolution import apply_kernel, gaussian_blur
from lumina.filters.basic import adjust_contrast, adjust_brightness

# set up a logger for this module
logger = logging.getLogger(__name__)


def dreamglow(image: Image, intensity: float = 0.6) -> Image:
    # create a soft glowing effect
    logger.info("applying dreamglow effect")

    # apply gaussian blur for the bloom effect
    blurred = gaussian_blur(image, size=11, sigma=3.0)

    # work in float so we don't overflow while blending
    base_data = image.data.astype(np.float32)
    glow_data = blurred.data.astype(np.float32)

    # apply screen blend mode
    base_norm = base_data / 255.0
    glow_norm = glow_data / 255.0

    blended = 1.0 - (1.0 - base_norm) * (1.0 - glow_norm * intensity)
    blended = blended * 255.0

    # restore contrast
    data = np.clip(blended, 0, 255).astype(np.uint8)
    img = Image(data)
    return adjust_contrast(img, 1.2)


def vignette(image: Image, strength: float = 1.5) -> Image:
    # darken the corners to draw attention to the center
    logger.info("applying vignette effect")

    h, w = image.data.shape[0], image.data.shape[1]

    # get coordinates from -1 to 1
    y, x = np.ogrid[-1 : 1 : h * 1j, -1 : 1 : w * 1j]

    # calculate distance from center
    radius = np.sqrt(x**2 + y**2)

    # create the mask
    # goes from 1 at the center to 0 at the edges based on strength
    mask = 1.0 - (radius * (strength / 1.5))
    mask = np.clip(mask, 0, 1)

    # reshape mask to broadcast across color channels
    if image.channels > 1:
        mask = mask[..., np.newaxis]

    # apply the mask to darken edges
    data = image.data.astype(np.float32) * mask
    data = np.clip(data, 0, 255).astype(np.uint8)

    return Image(data)


def vaporwave(image: Image) -> Image:
    # shift colors for a retro aesthetic
    # adjust color channels
    logger.info("applying vaporwave color shift")

    data = image.data.astype(np.float32)

    # only works for color images
    if image.channels >= 3:
        # adjust red channel
        data[:, :, 0] = data[:, :, 0] * 1.3 + 20
        # scale green channel
        data[:, :, 1] = data[:, :, 1] * 0.9
        # scale blue channel
        data[:, :, 2] = data[:, :, 2] * 1.4 + 40

    data = np.clip(data, 0, 255).astype(np.uint8)
    return Image(data)
