import os
import cv2

from value.strings import *
# utils
from utils.show_image import show_single_image
from utils.log import *

from algorithm import convert2gray

class Flocver:

    def __init__(self) -> None:
        self.images = []
        self.gray_images = []

    # make self.image
    def read_images(self) -> None:
        # just read the second dir
        for path in os.listdir(PATH_IMAGES):
            if os.path.isdir(os.path.join(PATH_IMAGES, path)):
                for path_ in os.listdir(os.path.join(PATH_IMAGES, path)):
                    file = os.path.join(os.path.join(PATH_IMAGES, path), path_)
                    if os.path.isfile(file):
                        self.images.append(cv2.imread(file))
                    else:
                        lw(f'{file} is not a file, can\'t be read as a picture')
        ln('read all pictures done')

    # make self.gray_images
    def convert2gray(self) -> None:
        for image in self.images:
            self.gray_images.append(convert2gray(image))
        ln('convert all image to gray')
    
    
