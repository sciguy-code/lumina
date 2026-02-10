import numpy as np

class Image:

    def __init__(self, data: np.ndarray):
        # crash early if the data is wrong.
        if not isinstance(data, np.ndarray):
            raise TypeError(f"image must be initialized with a NumPy array, got {type(data)}")
        
        self.data = data.astype(np.uint8)

    """
    [
        [ [R,G,B], [R,G,B] ],   <- row 0
        [ [R,G,B], [R,G,B] ]    <- row 1
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
        # if shape is (h,w) -> grayscale (1 channel)
        # if shape is (h,w,c) -> colored (c channel)
        if len(self.data.shape) == 2:
            return 1
        return self.data.shape[2]
    

    def __repr__(self):
        return f"<img {self.width}x{self.height} | {self.channels} ch"