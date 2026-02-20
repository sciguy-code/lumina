import argparse
import sys
import traceback

# io modules
from lumina.io.loader import load_image
from lumina.io.saver import save_image

# ops and filters
from lumina.ops.transform import to_grayscale
from lumina.filters.convolution import gaussian_blur
from lumina.filters.edges import sobel_filter
from lumina.ops.pooling import max_pool_2x2  

def entry_point():
    """
    main cli entry point for lumina.
    """
    parser = argparse.ArgumentParser(description="Lumina Image Engine")
    parser.add_argument("input_path", type=str, help="Path to input image")
    parser.add_argument("output_path", type=str, help="Path to save output")
    
    # cli flags
    parser.add_argument("--grayscale", action="store_true", help="Convert to grayscale")
    parser.add_argument("--blur", action="store_true", help="Apply Gaussian Blur")
    parser.add_argument("--edges", action="store_true", help="Apply Sobel Edge Detection")
    parser.add_argument("--pool", action="store_true", help="Apply 2x2 Max Pooling (Shrinks image by 50%%)") # max pooling flag

    args = parser.parse_args()

    try:
        print(f"[*] Loading {args.input_path}...")
        image = load_image(args.input_path)
        
        # processing pipeline
        
        # step 1: grayscale conversion
        # sobel works best on grayscale, so this is auto-applied for edge detection.
        if args.grayscale or args.edges:
            if image.channels == 3:
                print("[*] Converting to Grayscale...")
                image = to_grayscale(image)

        # step 2: blur
        # blur is commonly applied before edges to reduce noise.
        if args.blur:
            image = gaussian_blur(image)
            
        # step 3: edge detection
        if args.edges:
            image = sobel_filter(image)
            
        # step 4: max pooling
        # this reduces spatial size by 2x.
        if args.pool:
            image = max_pool_2x2(image)

        save_image(image, args.output_path)
        print(f"[*] Saved to {args.output_path}")

    except Exception as e:
        print(f"[!] Error: {e}")
        # print full traceback for debugging.
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    entry_point()
