from pathlib import Path
from PIL import Image as PILImage
import numpy as np
from lumina.core.image import Image


def load_image(path: str) -> Image:
    """
    reads an image file -> decodes it -> converts to NumPy -> wraps in Lumina Image.
    
    """
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"img not found: {path}")
    
    # using PIL as file reader 
    # converting to rgb -> we need 3 channels
    with PILImage.open(file_path) as img:
        img = img.convert("RGB")
        raw_data = np.array(img)

    return Image(raw_data)