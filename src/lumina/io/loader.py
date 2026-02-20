from pathlib import Path
from PIL import Image as PILImage
import numpy as np
from lumina.core.image import Image


def load_image(path: str) -> Image:
    """
    read an image file, decode it, convert to numpy, and wrap it in image.
    """
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"img not found: {path}")
    
    # use pil to read from disk.
    # convert to rgb so the output has three channels.
    with PILImage.open(file_path) as img:
        img = img.convert("RGB")
        raw_data = np.array(img)

    return Image(raw_data)
