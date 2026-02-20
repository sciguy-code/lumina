import numpy as np

class Image:

    def __init__(self, data: np.ndarray):
        # fail fast if input is not a numpy array.
        if not isinstance(data, np.ndarray):
            raise TypeError(f"image must be initialized with a NumPy array, got {type(data)}")
        
        self.data = data.astype(np.uint8)

    """
    pixel layout example:
    [
        [ [r,g,b], [r,g,b] ],   <- row 0
        [ [r,g,b], [r,g,b] ]    <- row 1
    ]

    """
    @property
    def height(self) -> int:
        return self.data.shape[0]
    

    @property
    def width(self) -> int:
        return self.data.shape[1]
    
    @property
    def channels(self) -> int:
        # shape (h, w) means grayscale (1 channel).
        # shape (h, w, c) means color image (c channels).
        if len(self.data.shape) == 2:
            return 1
        return self.data.shape[2]
    

    def __repr__(self):
        return f"<img {self.width}x{self.height} | {self.channels} ch"
