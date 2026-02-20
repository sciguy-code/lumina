import numpy as np
from lumina.core.image import Image
# this depends on the shared convolution utility.
from lumina.filters.convolution import apply_kernel

def sobel_filter(image: Image) -> Image:
    """
    detect edges with the sobel operator.
    compute gradient magnitude from vertical and horizontal responses.
    """
    print("[*] Applying Sobel Edge Detection...")

    # step 1: define sobel kernels.
    # kx emphasizes left-to-right intensity changes.
    Kx = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float32)

    # ky emphasizes top-to-bottom intensity changes.
    Ky = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ], dtype=np.float32)

    # step 2: run convolution in float32.
    # float32 preserves negative gradients during intermediate math.
    img_float_data = image.data.astype(np.float32)
    img_float_wrapper = Image(img_float_data)
    
    # compute vertical gradient response.
    gx_image = apply_kernel(img_float_wrapper, Kx)
    gx = gx_image.data.astype(np.float32)

    # compute horizontal gradient response.
    gy_image = apply_kernel(img_float_wrapper, Ky)
    gy = gy_image.data.astype(np.float32)

    # step 3: combine both directions into gradient magnitude.
    magnitude = np.sqrt(gx**2 + gy**2)

    # step 4: normalize so the strongest edge maps to 255.
    if magnitude.max() > 0:
        magnitude = (magnitude / magnitude.max()) * 255
    
    return Image(magnitude.astype(np.uint8))
