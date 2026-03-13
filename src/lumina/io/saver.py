import logging
from PIL import Image as PILImage
from lumina.core.image import Image

# set up a logger for this module
logger = logging.getLogger(__name__)

def save_image(image: Image, path: str) -> None:
    # convert our image back to a PIL image and save to disk
    # PIL figures out the format from the file extension which is convenient
    pil_img = PILImage.fromarray(image.data)
    pil_img.save(path)
    logger.info(f"saved image to {path}")
