from PIL import Image as PILImage
from lumina.core.image import Image

def save_image(image: Image, path: str):
    """
    take an image object, convert it to pil, and save it to disk.
    """
    # convert raw pixel data back into a pil image.
    pil_img = PILImage.fromarray(image.data)
    
    # save to disk; pil infers format from file extension.
    pil_img.save(path)
