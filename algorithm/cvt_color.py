import typing
import numpy as np

# input binary_image, cvt to other color
# change the background (if it black or white)
# color_from, for example 0 or 255
# color_to, for example (0, 255, 0) green
def cvt_color(binary_image: np.array, color_from: int, color_to: typing.Tuple[int], other: typing.Tuple[int] = (255, 255, 255)) -> np.array:
    # colored_image = np.zeros((binary_image.shape[0], binary_image.shape[1], 3), dtype=np.uint8)
    # colored_image[:] = color_to  
    new_image = np.zeros((binary_image.shape[0], binary_image.shape[1], 3), dtype=np.uint8)
    new_image[:] = other 

    new_image[binary_image == color_from] = color_to
    return new_image
    
