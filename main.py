from flocver import Flocver

from value.value import *
from qt.show_in_one_view import *
from utils.show_image import show_single_image

if __name__ == '__main__':
    flocver = Flocver()
    flocver.read_images()
    flocver.convert2gray()
    flocver.convert2binary()
    flocver.convert2morphology()
    flocver.convert2result(COLOR_FROM_DEFAULT, COLOR_TO_DEFAULT)
    flocver.show_result()



