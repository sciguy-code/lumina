import time
import numpy as np
from lumina.core.image import Image
from lumina.filters.convolution import gaussian_blur, build_gaussian_kernel, apply_kernel

# ----- blur benchmark -----
# this script tests how fast our gaussian blur is across different image sizes
# we also compare 2d kernel vs separable kernel if available

def run_blur_benchmark() -> None:
    # test these image sizes
    sizes = [256, 512, 1024, 2048]
    kernel_sizes = [3, 5, 7]

    print("=" * 60)
    print("GAUSSIAN BLUR BENCHMARK")
    print("=" * 60)
    print(f"{'Size':>8} | {'Kernel':>6} | {'Time (ms)':>10} | {'Mpx/s':>8}")
    print("-" * 60)

    for img_size in sizes:
        # generate a random grayscale image
        rng = np.random.RandomState(42)
        data = rng.randint(0, 256, (img_size, img_size), dtype=np.uint8)
        img = Image(data)

        for k_size in kernel_sizes:
            # warm up run to get rid of any initial overhead
            _ = gaussian_blur(img, size=k_size, sigma=1.0)

            # timed run - take the best of 3
            times = []
            for _ in range(3):
                start = time.perf_counter()
                _ = gaussian_blur(img, size=k_size, sigma=1.0)
                elapsed = time.perf_counter() - start
                times.append(elapsed)

            best_time = min(times)
            megapixels_per_sec = (img_size * img_size) / best_time / 1e6

            print(f"{img_size:>6}px | {k_size:>4}x{k_size} | {best_time*1000:>8.2f}ms | {megapixels_per_sec:>6.1f}")

    print("=" * 60)


def run_separable_vs_2d_benchmark() -> None:
    # compare separable 1d+1d convolution vs standard 2d convolution
    sizes = [256, 512, 1024]
    print("\n")
    print("=" * 60)
    print("SEPARABLE vs 2D KERNEL COMPARISON")
    print("=" * 60)
    print(f"{'Size':>8} | {'2D (ms)':>8} | {'Sep (ms)':>8} | {'Speedup':>8}")
    print("-" * 60)

    for img_size in sizes:
        rng = np.random.RandomState(42)
        data = rng.randint(0, 256, (img_size, img_size), dtype=np.uint8)
        img = Image(data)

        # build a 7x7 gaussian kernel (bigger kernel = bigger speedup for separable)
        kernel_2d = build_gaussian_kernel(7, 1.5)

        # for separable: decompose into two 1d kernels
        # a gaussian kernel is separable: K = col_vec * row_vec
        # we can get the 1d kernel from one row (or column) of the 2d kernel
        center_row = kernel_2d[3, :]
        # renormalize the 1d kernel
        kernel_1d = center_row / center_row.sum()

        # benchmark 2d convolution
        times_2d = []
        for _ in range(3):
            start = time.perf_counter()
            _ = apply_kernel(img, kernel_2d)
            elapsed = time.perf_counter() - start
            times_2d.append(elapsed)

        # benchmark separable (two 1d passes)
        times_sep = []
        for _ in range(3):
            start = time.perf_counter()
            # horizontal pass
            row_kernel = kernel_1d.reshape(1, -1)
            temp = apply_kernel(img, row_kernel)
            # vertical pass
            col_kernel = kernel_1d.reshape(-1, 1)
            _ = apply_kernel(temp, col_kernel)
            elapsed = time.perf_counter() - start
            times_sep.append(elapsed)

        best_2d = min(times_2d) * 1000
        best_sep = min(times_sep) * 1000
        speedup = best_2d / best_sep if best_sep > 0 else 0

        print(f"{img_size:>6}px | {best_2d:>6.2f}ms | {best_sep:>6.2f}ms | {speedup:>6.2f}x")

    print("=" * 60)


if __name__ == "__main__":
    run_blur_benchmark()
    run_separable_vs_2d_benchmark()
