import time
import numpy as np
from lumina.core.image import Image
from lumina.backends import get_backend

# ----- backend comparison benchmark -----
# compares numpy backend vs pure python backend
# spoiler: numpy is way faster but this shows exactly how much

def run_comparison() -> None:
    numpy_be = get_backend("numpy")
    python_be = get_backend("python")

    # using small images because the python backend is painfully slow
    sizes = [16, 32, 64, 128]

    print("=" * 70)
    print("NUMPY vs PYTHON BACKEND COMPARISON")
    print("=" * 70)

    # --- grayscale conversion ---
    print("\n--- Grayscale Conversion ---")
    print(f"{'Size':>8} | {'NumPy (ms)':>10} | {'Python (ms)':>12} | {'Speedup':>8}")
    print("-" * 50)

    for img_size in sizes:
        rng = np.random.RandomState(42)
        data = rng.randint(0, 256, (img_size, img_size, 3), dtype=np.uint8)
        img = Image(data)

        # time numpy backend
        start = time.perf_counter()
        _ = numpy_be.to_grayscale(img)
        np_time = time.perf_counter() - start

        # time python backend
        start = time.perf_counter()
        _ = python_be.to_grayscale(img)
        py_time = time.perf_counter() - start

        speedup = py_time / np_time if np_time > 0 else 0
        print(f"{img_size:>6}px | {np_time*1000:>8.3f}ms | {py_time*1000:>10.3f}ms | {speedup:>6.1f}x")

    # --- convolution (3x3 mean kernel) ---
    print("\n--- 3x3 Convolution ---")
    print(f"{'Size':>8} | {'NumPy (ms)':>10} | {'Python (ms)':>12} | {'Speedup':>8}")
    print("-" * 50)

    kernel = np.ones((3, 3), dtype=np.float32) / 9

    for img_size in sizes:
        rng = np.random.RandomState(42)
        data = rng.randint(0, 256, (img_size, img_size), dtype=np.uint8)
        img = Image(data)

        start = time.perf_counter()
        _ = numpy_be.apply_kernel(img, kernel)
        np_time = time.perf_counter() - start

        start = time.perf_counter()
        _ = python_be.apply_kernel(img, kernel)
        py_time = time.perf_counter() - start

        speedup = py_time / np_time if np_time > 0 else 0
        print(f"{img_size:>6}px | {np_time*1000:>8.3f}ms | {py_time*1000:>10.3f}ms | {speedup:>6.1f}x")

    # --- max pooling ---
    print("\n--- 2x2 Max Pooling ---")
    print(f"{'Size':>8} | {'NumPy (ms)':>10} | {'Python (ms)':>12} | {'Speedup':>8}")
    print("-" * 50)

    for img_size in sizes:
        rng = np.random.RandomState(42)
        data = rng.randint(0, 256, (img_size, img_size), dtype=np.uint8)
        img = Image(data)

        start = time.perf_counter()
        _ = numpy_be.max_pool(img)
        np_time = time.perf_counter() - start

        start = time.perf_counter()
        _ = python_be.max_pool(img)
        py_time = time.perf_counter() - start

        speedup = py_time / np_time if np_time > 0 else 0
        print(f"{img_size:>6}px | {np_time*1000:>8.3f}ms | {py_time*1000:>10.3f}ms | {speedup:>6.1f}x")

    # --- invert ---
    print("\n--- Color Inversion ---")
    print(f"{'Size':>8} | {'NumPy (ms)':>10} | {'Python (ms)':>12} | {'Speedup':>8}")
    print("-" * 50)

    for img_size in sizes:
        rng = np.random.RandomState(42)
        data = rng.randint(0, 256, (img_size, img_size, 3), dtype=np.uint8)
        img = Image(data)

        start = time.perf_counter()
        _ = numpy_be.invert(img)
        np_time = time.perf_counter() - start

        start = time.perf_counter()
        _ = python_be.invert(img)
        py_time = time.perf_counter() - start

        speedup = py_time / np_time if np_time > 0 else 0
        print(f"{img_size:>6}px | {np_time*1000:>8.3f}ms | {py_time*1000:>10.3f}ms | {speedup:>6.1f}x")

    print("\n" + "=" * 70)
    print("(python backend is slower but shows the algorithm in plain loops)")
    print("=" * 70)


if __name__ == "__main__":
    run_comparison()
