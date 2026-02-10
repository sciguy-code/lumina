from PIL import Image as PILImage
from lumina.core.image import Image

def save_image(image: Image, path: str):
    """
    Takes a Lumina Image -> Converts to PIL -> Encodes to format -> Saves to Disk.
    """
    # Convert our raw matrix back to a PIL object
    pil_img = PILImage.fromarray(image.data)
    
    # Save it. PIL infers format (png/jpg) from the filename extension.
    pil_img.save(path)