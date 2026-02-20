import numpy as np
from lumina.core.image import Image

def to_grayscale(image : Image) -> Image:
    
    # luma formula: gray = 0.299r + 0.587g + 0.114b

    if image.channels == 1:
        print("img already grayscaled")
        return Image
    
    weights = np.array([0.229, 0.587, 0.114])

    grayscale_data = np.dot(image.data[..., :3], weights)

    grayscale_data = grayscale_data.astype(np.uint8)

    return Image(grayscale_data)
