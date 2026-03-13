import numpy as np

class Image:
    # this is basically a wrapper around a numpy array that knows it's an image
    # it keeps track of height, width, channels so we don't have to remember shape indices

    def __init__(self, data: np.ndarray) -> None:
        # fail fast if someone passes in something weird
        if not isinstance(data, np.ndarray):
            raise TypeError(f"image must be initialized with a NumPy array, got {type(data)}")

        self.data: np.ndarray = data.astype(np.uint8)

    # pixel layout reminder:
    # [
    #     [ [r,g,b], [r,g,b] ],   <- row 0
    #     [ [r,g,b], [r,g,b] ]    <- row 1
    # ]

    @property
    def height(self) -> int:
        return int(self.data.shape[0])

    @property
    def width(self) -> int:
        return int(self.data.shape[1])

    @property
    def channels(self) -> int:
        # shape (h, w) means grayscale, shape (h, w, c) means color
        if len(self.data.shape) == 2:
            return 1
        return int(self.data.shape[2])

    def __repr__(self) -> str:
        return f"<img {self.width}x{self.height} | {self.channels} ch>"
