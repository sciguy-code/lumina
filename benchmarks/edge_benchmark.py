import time
import numpy as np
from lumina.core.image import Image
from lumina.filters.edges import sobel_filter
from lumina.ops.transform import to_grayscale

# ----- edge detection benchmark -----
# tests how fast sobel edge detection runs on different image sizes

def run_edge_benchmark() -> None:
    sizes = [256, 512, 1024, 2048]

    print("=" * 60)
    print("SOBEL EDGE DETECTION BENCHMARK")
    print("=" * 60)
    print(f"{'Size':>8} | {'Time (ms)':>10} | {'Mpx/s':>8}")
    print("-" * 60)

    for img_size in sizes:
        # generate a random grayscale image (sobel works on grayscale)
        rng = np.random.RandomState(42)
        data = rng.randint(0, 256, (img_size, img_size), dtype=np.uint8)
        img = Image(data)

        # warm up
        _ = sobel_filter(img)

        # timed runs - best of 3
        times = []
        for _ in range(3):
            start = time.perf_counter()
            _ = sobel_filter(img)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        best_time = min(times)
        megapixels_per_sec = (img_size * img_size) / best_time / 1e6

        print(f"{img_size:>6}px | {best_time*1000:>8.2f}ms | {megapixels_per_sec:>6.1f}")

    print("=" * 60)


if __name__ == "__main__":
    run_edge_benchmark()
