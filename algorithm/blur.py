import cv2
import numpy

def cvt_gaussain_blur(image: numpy.array) -> numpy.array:
    # (5, 5) in represents the size of the Gaussian kernel, 
    # that is, the width and height of the convolution kernel. 
    # The 5 here represents a 5x5 convolution kernel.
    return cv2.GaussianBlur(image, (5, 5), 0)

def cvt_mean_blur(image: numpy.array) -> numpy.array:
    # (5, 5) represents the size of the mean filter kernel, 
    # that is, the width and height of the convolution kernel. 
    # The 5 here represents a 5x5 convolution kernel.
    return cv2.blur(image, (5,5))

def cvt_bilateral_blur(image: numpy.array) -> numpy.array:
    return cv2.bilateralFilter(image, 9, 75, 75)


