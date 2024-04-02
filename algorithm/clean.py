import cv2
import numpy as np

from value.value import *

def morphology(binary_image) -> np.array:
    morphology_kernel = np.ones(MORPHOLOGY_KERNEL_SIZE, np.uint8)
    cleaned_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, morphology_kernel)

    return cleaned_image
