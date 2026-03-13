from abc import ABC, abstractmethod
import numpy as np
from lumina.core.image import Image


class Backend(ABC):
    # this is the abstract base class that all backends must implement
    # the idea is that we can swap between numpy and pure python
    # depending on what's available or what the user wants

    @abstractmethod
    def apply_kernel(self, image: Image, kernel: np.ndarray) -> Image:
        # apply a convolution kernel to an image
        ...

    @abstractmethod
    def max_pool(self, image: Image) -> Image:
        # do 2x2 max pooling
        ...

    @abstractmethod
    def to_grayscale(self, image: Image) -> Image:
        # convert rgb to grayscale
        ...

    @abstractmethod
    def adjust_brightness(self, image: Image, factor: float) -> Image:
        # scale pixel intensity
        ...

    @abstractmethod
    def adjust_contrast(self, image: Image, factor: float) -> Image:
        # adjust contrast around the mean
        ...

    @abstractmethod
    def sharpen(self, image: Image) -> Image:
        # apply sharpening filter
        ...

    @abstractmethod
    def invert(self, image: Image) -> Image:
        # invert pixel values
        ...
