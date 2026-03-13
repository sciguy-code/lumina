# lumina

a high-performance, vectorized image processing engine built from scratch in python. implements core computer vision operations like convolution and filtering with optimized numpy routines instead of high-level library calls.

## sample outputs

all outputs below were generated from a single input image using the lumina cli.

### original

![original](test.jpg)

---

### grayscale

```bash
lumina-cli test.jpg samples/grayscale.png --grayscale
```

![grayscale](samples/grayscale.png)

---

### gaussian blur (3×3, σ=1.0)

```bash
lumina-cli test.jpg samples/blur.png --blur
```

![blur 3x3](samples/blur.png)

---

### gaussian blur (7×7, σ=2.0)

```bash
lumina-cli test.jpg samples/blur_large.png --blur --blur-size 7 --sigma 2.0
```

![blur 7x7](samples/blur_large.png)

---

### sobel edge detection

```bash
lumina-cli test.jpg samples/edges.png --edges
```

![edges](samples/edges.png)

---

### sharpen

```bash
lumina-cli test.jpg samples/sharpen.png --sharpen
```

![sharpen](samples/sharpen.png)

---

### brightness (factor 1.5)

```bash
lumina-cli test.jpg samples/bright.png --brightness 1.5
```

![brightness](samples/bright.png)

---

### contrast (factor 1.5)

```bash
lumina-cli test.jpg samples/contrast.png --contrast 1.5
```

![contrast](samples/contrast.png)

---

### invert

```bash
lumina-cli test.jpg samples/invert.png --invert
```

![invert](samples/invert.png)

---

### 2×2 max pooling

```bash
lumina-cli test.jpg samples/pooled.png --pool
```

![pooled](samples/pooled.png)

---

### full pipeline (grayscale → blur → edges → pool)

```bash
lumina-cli test.jpg samples/full_pipeline.png --blur --edges --pool --verbose
```

![full pipeline](samples/full_pipeline.png)

---

## what it does

- loads images from disk (`pillow`), stores pixel data as `numpy` arrays
- supports grayscale conversion (bt.601 luma weights)
- supports gaussian blur (configurable kernel size and sigma)
- supports sobel edge detection
- supports 2×2 max pooling
- supports brightness, contrast, sharpen, and invert filters
- chainable pipeline engine for composing operations
- dual backend: vectorized numpy (fast) and pure python (educational)
- saves output images back to disk

## install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

for development (includes pytest, mypy):

```bash
pip install -e ".[dev]"
```

## cli usage

```bash
lumina-cli <input_path> <output_path> [flags]
```

### available flags

| flag | description |
|---|---|
| `--grayscale` | convert to grayscale |
| `--blur` | apply gaussian blur |
| `--blur-size N` | kernel size for blur (must be odd, default: 3) |
| `--sigma F` | sigma for gaussian blur (default: 1.0) |
| `--edges` | apply sobel edge detection |
| `--pool` | apply 2×2 max pooling (shrinks image by 50%) |
| `--sharpen` | apply sharpening filter |
| `--brightness F` | adjust brightness by factor (e.g. 1.5 = brighter) |
| `--contrast F` | adjust contrast by factor (e.g. 1.3 = more contrast) |
| `--invert` | invert image colors |
| `--verbose` | show debug-level logging output |

### pipeline order

1. grayscale (auto-applied when `--edges` is used on rgb input)
2. brightness (optional)
3. contrast (optional)
4. blur (optional)
5. sharpen (optional)
6. edge detection (optional)
7. invert (optional)
8. max pooling (optional)

### examples

```bash
# basic edge detection
lumina-cli test.jpg output.png --edges

# full pipeline with verbose logging
lumina-cli test.jpg output.png --blur --edges --pool --verbose

# strong blur with custom kernel
lumina-cli test.jpg output.png --blur --blur-size 9 --sigma 3.0

# brightness + contrast combo
lumina-cli test.jpg output.png --brightness 1.3 --contrast 1.5
```

## pipeline engine

you can also use lumina programmatically with the pipeline engine:

```python
from lumina.io.loader import load_image
from lumina.io.saver import save_image
from lumina.ops.transform import to_grayscale
from lumina.filters.convolution import gaussian_blur
from lumina.filters.edges import sobel_filter
from lumina.pipeline.engine import Pipeline

image = load_image("test.jpg")

result = (
    Pipeline()
    .add(to_grayscale)
    .add(gaussian_blur, size=5, sigma=1.5)
    .add(sobel_filter)
    .run(image)
)

save_image(result, "output.png")
```

## backends

lumina has two backends that implement the same operations:

- **numpy** (default) — vectorized array operations, fast
- **python** — pure nested loops, slow but readable

```python
from lumina.backends import get_backend

fast = get_backend("numpy")
slow = get_backend("python")

result_fast = fast.to_grayscale(image)
result_slow = slow.to_grayscale(image)
# both produce the same output (within rounding tolerance)
```

## project structure

```text
src/lumina/
├── cli/main.py                # cli entry point
├── core/image.py              # image container class
├── io/
│   ├── loader.py              # image loading (pillow)
│   └── saver.py               # image saving
├── filters/
│   ├── convolution.py         # kernel convolution + gaussian blur
│   ├── edges.py               # sobel edge detection
│   └── basic.py               # brightness, contrast, sharpen, invert
├── ops/
│   ├── transform.py           # grayscale conversion
│   └── pooling.py             # 2×2 max pooling
├── pipeline/engine.py         # chainable pipeline
└── backends/
    ├── base.py                # abstract backend interface
    ├── numpy_backend.py       # vectorized backend
    └── python.py              # pure python backend

tests/                         # 90 tests across 10 files
benchmarks/                    # blur, edge, backend comparison
samples/                       # generated output images
```

## testing

```bash
# run all tests
pytest tests/ -v

# run with coverage
pytest tests/ -v --cov=lumina --cov-report=term-missing

# type checking
mypy src/lumina/ --ignore-missing-imports
```

**current status:** 90 tests passing, 76% coverage, 0 mypy errors.

## benchmarks

```bash
python benchmarks/blur_benchmark.py
python benchmarks/edge_benchmark.py
python benchmarks/backend_comparison.py
```

## tech stack

- python `>=3.9`
- `numpy` — vectorized array operations
- `pillow` — image file i/o
- `pytest` + `pytest-cov` — testing
- `mypy` — static type checking
