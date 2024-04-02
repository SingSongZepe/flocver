import cv2
import numpy as np

from value.value import *

# binary image is recommended
def canny(binary_image: np.array) -> np.array:
    # return edges
    return cv2.Canny(binary_image, CANNY_THRESHOLD_1, CANNY_THRESHOLD_2)

# edges expansion
def dilate(edges: np.array) -> np.array:
    dilated_kernel = np.ones(DILATED_KERNEL_SIZE, np.uint8)
    # return dilated_edges
    return cv2.dilate(edges, dilated_kernel, iterations=DILATED_ITERATIONS)



