import cv2
import numpy as np

from value.value import *

# binary image is recommended
def canny(binary_image: np.array) -> np.array:
    # return edges
    return cv2.Canny(binary_image, CANNY_THRESHOLD_1, CANNY_THRESHOLD_2)

# binary_image dilation
def dilate(binary_image: np.array) -> np.array:
    dilated_kernel = np.ones(DILATED_KERNEL_SIZE, np.uint8)
    # return dilated_edges
    return cv2.dilate(binary_image, dilated_kernel, iterations=DILATED_ITERATIONS)

# binary_image erosion
def erode(binary_image: np.array) -> np.array:
    eroded_kernel = np.ones(ERODED_KERNEL_SIZE, np.uint8)
    eroded_image = cv2.erode(binary_image, eroded_kernel, iterations=ERODED_ITERATIONS)
    return eroded_image
