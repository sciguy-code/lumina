import argparse
import sys
from lumina.io.loader import load_image
from lumina.io.saver import save_image
from lumina.ops.transform import to_grayscale

def entry_point():
    parser = argparse.ArgumentParser(description="Lumina Image Engine")
    parser.add_argument("input_path", type=str, help="Path to the input image")
    parser.add_argument("output_path", type=str, help="Path to save the output image")

    parser.add_argument("--grayscale", action="store_true", help="Convert to grayscale")

    args = parser.parse_args()

    try:
        print(f"[*] Loading {args.input_path}...")
        image = load_image(args.input_path)
        print(f"[*] Successfully loaded: {image}")

        # --- image transformation here ---
        if args.grayscale:
            print("[*] Applying Grayscale (Vectorized)...")
            image = to_grayscale(image)
        # -------------------------------------------

        save_image(image, args.output_path)
        print(f"[*] Saved to {args.output_path}")

    except Exception as e:
        print(f"[!] Error: {e}")
        sys.exit(1)