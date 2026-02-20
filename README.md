# lumina

lumina is a small image-processing engine built in python with a cli.

## what it does right now

- loads images from disk (`pillow`), stores pixel data as `numpy` arrays
- supports grayscale conversion
- supports gaussian blur (`3x3` kernel)
- supports sobel edge detection
- supports `2x2` max pooling
- saves output images back to disk

## install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## cli usage

```bash
lumina-cli <input_path> <output_path> [--grayscale] [--blur] [--edges] [--pool]
```

example:

```bash
lumina-cli test.jpg pipeline.png --blur --edges --pool
```

pipeline order:

1. grayscale (auto-applied when `--edges` is used on rgb input)
2. blur (optional)
3. edge detection (optional)
4. max pooling (optional)

## project structure

```text
src/lumina/cli/main.py           # cli entry point
src/lumina/io/loader.py          # image loading
src/lumina/io/saver.py           # image saving
src/lumina/filters/convolution.py # kernel application + gaussian blur
src/lumina/filters/edges.py      # sobel filter
src/lumina/ops/transform.py      # grayscale transform
src/lumina/ops/pooling.py        # 2x2 max pooling
src/lumina/core/image.py         # image container class
```

## tech stack

- python `>=3.9`
- `numpy`
- `pillow`

## future checklist

- [ ] add real test coverage for core, filters, and cli
- [ ] fix grayscale weight constant in `to_grayscale` (`0.299` instead of `0.229`)
- [ ] fix grayscale early return in `to_grayscale` to return the input image instance
- [ ] add benchmark scripts for blur and edge detection
- [ ] add examples for each cli flag combination
- [ ] add support for configurable kernel sizes from cli
- [ ] add package release workflow (build + publish)
