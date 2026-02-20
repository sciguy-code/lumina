import numpy as np
from lumina.core.image import Image

def max_pool_2x2(image: Image) -> Image:
    """
    reduce image dimensions by half using 2x2 max pooling.
    use reshaping to avoid explicit python loops.
    """
    print("[*] Applying 2x2 Max Pooling...")
    
    data = image.data
    is_grayscale = (data.ndim == 2)
    if is_grayscale:
        data = data[:, :, None]

    # step 1: read input dimensions.
    h, w, c = data.shape
    
    # step 2: trim edges so height and width are divisible by 2.
    # example: width 1001 becomes 1000 by dropping the last column.
    new_h = h - (h % 2)
    new_w = w - (w % 2)
    trimmed_data = data[:new_h, :new_w, :]
    
    # step 3: reshape into non-overlapping 2x2 windows.
    # (h, w, c) becomes (h/2, 2, w/2, 2, c).
    reshaped = trimmed_data.reshape(new_h // 2, 2, new_w // 2, 2, c)
    
    # step 4: take the max over each 2x2 block.
    # axis=1 and axis=3 correspond to the two pooled dimensions.
    pooled_data = reshaped.max(axis=(1, 3))
    
    if is_grayscale:
        pooled_data = pooled_data[:, :, 0]

    return Image(pooled_data.astype(np.uint8))
