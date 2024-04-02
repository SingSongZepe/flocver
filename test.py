import cv2
import numpy as np
import typing

from value.value import *
from value.strings import *
from utils.show_image import show_single_image
from algorithm.blur import *
from error.size_not_match import SizeNotMatch

def main(path_pic: str):
    image = cv2.imread(path_pic)

    # turn to gray
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, binary_image = cv2.threshold(gray_image, BINARY_THRESHOLD, 255, cv2.THRESH_BINARY)
    
    show_single_image(binary_image)


# Otsu's algorithm
def otsu(path_pic: str) -> numpy.array:
    image = cv2.imread(path_pic)

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    blur_image = cvt_bilateral_blur(binary_image)
    # show_single_image(blur_image)
    return blur_image

# canny
def canny(path_pic: str) -> numpy.array:
    binary_image = otsu(path_pic)
    edges = cv2.Canny(binary_image, 100, 200)

    # show_single_image(edges)
    return edges
    
# dilate
def dilate(path_pic: str):
    edges = canny(path_pic)

    dilated_kernel = np.ones((5, 5), np.uint8)
    dilated_edges = cv2.dilate(edges, dilated_kernel, iterations=5) 

    # show_single_image(dilated_edges)
    return dilated_edges

# morphologyEx
def morphology(binary_image) -> np.array:

    # there, range from 10 to 18
    kernel = np.ones((10, 10), np.uint8)
    cleaned_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)

    # show_single_image(cleaned_image)
    return cleaned_image

# binary for black to green
def black2green(binary_image: np.array) -> np.array:
    colored_image = np.zeros((binary_image.shape[0], binary_image.shape[1], 3), dtype=np.uint8)
    colored_image[:] = (0, 255, 0)  

    colored_image[binary_image == 255] = (255, 255, 255)

    return colored_image

# overlap
# The first binary image (white 255) covers the second one (may be not binary)
def overlap(i1: np.array, i2: np.array) -> np.array:
    if i1.shape[0] != i2.shape[0] or i1.shape[1] != i2.shape[1]:
        raise SizeNotMatch('Error, size of image 1 not matches image 2')

    i2[i1 == 255] = (255, 255, 255)

    return i2
    

# test algorithm
if __name__ == '__main__':
    # otsu(PATH_IMAGE_EASY_1)
    # otsu(PATH_IMAGE_MEDIUM_2)
    # otsu(PATH_IMAGE_HARD_1)
    # canny(PATH_IMAGE_EASY_1)
    dilated_image = morphology(dilate(PATH_IMAGE_EASY_1))
    show_single_image(dilated_image)
    bi = morphology(otsu(PATH_IMAGE_EASY_1))
    bi_with_green = black2green(bi)
    show_single_image(bi_with_green)
    
    overlapped_image = overlap(dilated_image, bi_with_green)
    show_single_image(overlapped_image)