import argparse
import logging
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
from lumina.filters.basic import adjust_brightness, adjust_contrast, sharpen, invert

# set up a logger for the cli
logger = logging.getLogger(__name__)

def entry_point() -> None:
    # main cli entry point for lumina
    parser = argparse.ArgumentParser(description="Lumina Image Engine")
    parser.add_argument("input_path", type=str, help="Path to input image")
    parser.add_argument("output_path", type=str, help="Path to save output")

    # basic operations
    parser.add_argument("--grayscale", action="store_true", help="Convert to grayscale")
    parser.add_argument("--blur", action="store_true", help="Apply Gaussian Blur")
    parser.add_argument("--edges", action="store_true", help="Apply Sobel Edge Detection")
    parser.add_argument("--pool", action="store_true", help="Apply 2x2 Max Pooling (shrinks image by 50%%)")

    # blur config
    parser.add_argument("--blur-size", type=int, default=3, help="Kernel size for blur (must be odd, default: 3)")
    parser.add_argument("--sigma", type=float, default=1.0, help="Sigma for gaussian blur (default: 1.0)")

    # basic filter operations
    parser.add_argument("--brightness", type=float, default=None, help="Adjust brightness by factor (e.g. 1.5 = brighter)")
    parser.add_argument("--contrast", type=float, default=None, help="Adjust contrast by factor (e.g. 1.3 = more contrast)")
    parser.add_argument("--sharpen", action="store_true", help="Apply sharpening filter")
    parser.add_argument("--invert", action="store_true", help="Invert image colors")

    # logging control
    parser.add_argument("--verbose", action="store_true", help="Show debug-level logging output")

    args = parser.parse_args()

    # set up logging based on verbosity flag
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="[%(levelname)s] %(name)s: %(message)s"
    )

    try:
        logger.info(f"loading {args.input_path}")
        image = load_image(args.input_path)

        # --- processing pipeline ---
        # the order here matters, we go:
        # grayscale -> brightness/contrast -> blur -> sharpen -> edges -> invert -> pool

        # step 1: grayscale conversion
        # sobel works best on grayscale so we auto-apply it for edge detection
        if args.grayscale or args.edges:
            if image.channels == 3:
                logger.info("converting to grayscale")
                image = to_grayscale(image)

        # step 2: brightness adjustment
        if args.brightness is not None:
            image = adjust_brightness(image, args.brightness)

        # step 3: contrast adjustment
        if args.contrast is not None:
            image = adjust_contrast(image, args.contrast)

        # step 4: blur
        if args.blur:
            image = gaussian_blur(image, size=args.blur_size, sigma=args.sigma)

        # step 5: sharpen
        if args.sharpen:
            image = sharpen(image)

        # step 6: edge detection
        if args.edges:
            image = sobel_filter(image)

        # step 7: invert
        if args.invert:
            image = invert(image)

        # step 8: max pooling
        if args.pool:
            image = max_pool_2x2(image)

        save_image(image, args.output_path)
        logger.info(f"saved to {args.output_path}")

    except Exception as e:
        logger.error(f"error: {e}")
        # print full traceback so we can actually debug stuff
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    entry_point()
