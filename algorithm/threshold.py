import cv2
import numpy as np

from value.value import *

# input, gray_images are recommended
def common_threshold(gray_image: np.array, thread: int) -> np.array:
    _, binar_image = cv2.threshold(gray_image, thread, 255, cv2.THRESH_BINARY)
    return binar_image

def otsu_threshold(gray_image: np.array) -> np.array:
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_image
    
