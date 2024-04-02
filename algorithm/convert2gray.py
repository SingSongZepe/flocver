import cv2
import numpy

def convert2gray(image: numpy.array) -> numpy.array:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def convert2(image: numpy.array, flag) -> numpy.array:
    return cv2.cvtColor(image, flag)
