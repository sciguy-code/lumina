import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from lumina.core.image import Image

def apply_kernel(image: Image, kernel: np.ndarray) -> Image:
    """
    apply a kernel over an image using vectorized sliding windows.
    this is the core operation for blur, sharpen, and edge filters.
    """
    # step 1: read input dimensions
    k_h, k_w = kernel.shape
    data = image.data
    is_grayscale = (data.ndim == 2)
    if is_grayscale:
        # promote grayscale data to (h, w, 1) so the same path works for all images.
        data = data[:, :, None]
    h, w, c = data.shape
    
    # step 2: compute edge padding so the kernel can cover border pixels.
    pad_h = k_h // 2
    pad_w = k_w // 2
    
    # mode='edge' repeats border values, which avoids dark borders.
    # pad only height and width, not channels.
    padded_image = np.pad(data, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='edge')
    
    # step 3: build sliding windows for each kernel-sized patch.
    # this creates a 5d view: (height, width, channels, kernel_h, kernel_w).
    windows = sliding_window_view(padded_image, (k_h, k_w), axis=(0, 1))
    
    # step 4: apply kernel weights and reduce across kernel dimensions.
    output_data = np.sum(windows * kernel, axis=(3, 4))
    if is_grayscale:
        # squeeze back to (h, w) for grayscale output.
        output_data = output_data[:, :, 0]
    
    # step 5: clamp to valid pixel range and cast to uint8.
    output_data = np.clip(output_data, 0, 255).astype(np.uint8)
    
    return Image(output_data)

def gaussian_blur(image: Image, size: int = 3) -> Image:
    """
    build a gaussian-like kernel and apply it to the image.
    """
    print(f"[*] Applying {size}x{size} Gaussian Blur...")
    
    if size == 3:
        # standard 3x3 gaussian approximation.
        kernel = np.array([
            [1, 2, 1],
            [2, 4, 2],
            [1, 2, 1]
        ]) / 16.0  # normalize so overall brightness is preserved.
    else:
        # fallback to a simple mean kernel for other sizes.
        kernel = np.ones((size, size)) / (size * size)

    return apply_kernel(image, kernel)
