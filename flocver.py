import os
import cv2
from os.path import join as j

from value.strings import *
# utils
from utils.show_image import show_single_image
from utils.log import *
from utils.make_dir import make_dir
from utils.save_file import save_file
from qt.show_in_one_view import show_in_one_view

from algorithm.blur import *
from algorithm.convert2gray import *
from algorithm.threshold import *
from algorithm.clean import *
from algorithm.cvt_color import *
from algorithm.edge import *

class Flocver:

    def __init__(self) -> None:
        self.images = []
        self.gray_images = []
        self.binary_images = []
        self.cleaned_images = []
        self.result_images = []

    # make self.image
    def read_images(self) -> None:
        # just read the second dir
        for path in os.listdir(PATH_IMAGES):
            if os.path.isdir(j(PATH_IMAGES, path)):
                for path_ in os.listdir(j(PATH_IMAGES, path)):
                    file = j(j(PATH_IMAGES, path), path_)
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

    # make self.binary_images
    def convert2binary(self) -> None:
        for idx, image in enumerate(self.gray_images):
            # there you can select another algorithm common_threshold
            # but you need to select a suitable threshold value
            if idx == 0:
                self.binary_images.append(common_threshold(image, BINARY_THRESHOLD_EASY_1))
            elif idx == 4:
                self.binary_images.append(common_threshold(image, BINARY_THRESHOLD_HARD_2))
            elif idx == 6:
                self.binary_images.append(common_threshold(image, BINARY_THRESHOLD_MEDIUM_1))
            elif idx == 8:
                self.binary_images.append(common_threshold(image, BINARY_THRESHOLD_MEDIUM_3))
            else:
                self.binary_images.append(otsu_threshold(image))
        ln('convert all image to binary')
    
    # make self.cleaned_images
    def convert2morphology(self) -> None:
        for image in self.binary_images:
            # there you can select another algorithm blur
            # but it is not as suitable as algorithm morphology to do it
            self.cleaned_images.append(morphology(image))
        ln('convert all image to cleaned')

    # make self.result_images
    def convert2result(self, color_from: int, color_to: typing.Tuple[int, 3]) -> None:
        for image in self.cleaned_images:
            self.result_images.append(cvt_color(image, color_from, color_to))
        ln('convert all image to result')

    def show_result(self):
        show_in_one_view(self.result_images)

    def save_result(self):
        # check directory
        easy_output_path = j(PATH_OUTPUT, EASY_)
        medium_output_path = j(PATH_OUTPUT, MEDIUM_)
        hard_output_path = j(PATH_OUTPUT, HARD_)
        make_dir(easy_output_path)
        make_dir(medium_output_path)
        make_dir(hard_output_path)
        
        # easy
        for idx, image in enumerate(self.result_images[:3]):
            save_file(image, j(easy_output_path, EASY_ + f'_{idx + 1}.jpg'))
        # medium
        for idx, image in enumerate(self.result_images[3:6]):
            save_file(image, j(hard_output_path, HARD_ + f'_{idx + 1}.jpg'))
        # hard
        for idx, image in enumerate(self.result_images[6:]):
            save_file(image, j(medium_output_path, MEDIUM_ + f'_{idx + 1}.jpg'))
            
