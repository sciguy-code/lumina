import logging
from pathlib import Path
from PIL import Image as PILImage
import numpy as np
from lumina.core.image import Image

# set up a logger for this module
logger = logging.getLogger(__name__)

def load_image(path: str) -> Image:
    # read an image file from disk and wrap it in our Image class
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"image not found: {path}")

    # use pillow to handle the actual file decoding
    # convert to RGB so we always get 3 channels to work with
    with PILImage.open(file_path) as img:
        rgb_img = img.convert("RGB")
        raw_data = np.array(rgb_img)

    logger.info(f"loaded image from {path} ({raw_data.shape[1]}x{raw_data.shape[0]})")
    return Image(raw_data)
