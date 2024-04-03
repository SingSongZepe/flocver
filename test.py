import cv2
import numpy as np
import typing

from PySide6.QtWidgets import QApplication, QLabel, QGridLayout, QWidget
from PySide6.QtGui import QPixmap, QImage

from value.value import *
from value.strings import *
from utils.show_image import show_single_image
from qt.show_in_one_view import show_in_one_view
from algorithm.blur import *
from algorithm.cvt_color import *
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
    dilated_edges = cv2.dilate(edges, dilated_kernel, iterations=4) 

    # show_single_image(dilated_edges)
    return dilated_edges

# morphologyEx
def morphology(binary_image) -> np.array:

    # there, range from 10 to 18
    kernel = np.ones((20, 20), np.uint8)
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

# fill
def fill(binary_image: np.array) -> np.array:
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros_like(binary_image)

    cv2.drawContours(mask, contours, -1, (255, 255, 255), thickness=cv2.FILLED)

    return cv2.bitwise_and(binary_image, mask)

# test algorithm
if __name__ == '__main__':
    # otsu(PATH_IMAGE_EASY_1)
    # otsu(PATH_IMAGE_MEDIUM_2)
    # otsu(PATH_IMAGE_HARD_1)
    # canny(PATH_IMAGE_EASY_1)
    # dilated_image = morphology(dilate(PATH_IMAGE_EASY_1))
    # # show_single_image(dilated_image)
    # bi = morphology(otsu(PATH_IMAGE_HARD_1))
    # bi_with_green = black2green(bi)
    # # show_single_image(bi_with_green)
    
    # # overlapped_image = overlap(dilated_image, bi_with_green)
    # # show_single_image(overlapped_image)

    # a = cvt_color(bi, 0, (0, 255, 0))
    # show_single_image(a)
    
    # easy 1
    ie1 = morphology(otsu(PATH_IMAGE_EASY_1))
    ce1 = cvt_color(ie1, 0, (0, 255, 0))
    # show_single_image(ce1)

    # medium 2
    im2 = morphology(otsu(PATH_IMAGE_MEDIUM_2))
    cm2 = cvt_color(im2, 0, (0, 255, 0))
    # show_single_image(cm2)

    # hard 2
    ih2 = morphology(otsu(PATH_IMAGE_HARD_2))
    ah2 = fill(ih2)
    ch2 = cvt_color(ah2, 0, (0, 255, 0))
    show_single_image(ch2)
    
    # hard 3
    ih3 = morphology(otsu(PATH_IMAGE_HARD_3))
    ch3 = cvt_color(ih3, 0, (0, 255, 0))
    # show_single_image(ch3)

    # images = [ce1, cm2, ch3, ce1, cm2, ch3, ce1, cm2, ch3]

    # show_in_one_view(images)

    # horizontal_concatenated = cv2.hconcat([ce1, cm2, ch3])
    # show_single_image(horizontal_concatenated)

    # test for qt (pyside6)
    # app = QApplication([])
    # wgt = QWidget()
    # gl = QGridLayout()
    # lb1 = QLabel()
    # lb2 = QLabel()
    # lb3 = QLabel()

    # height, width, channel = ce1.shape
    # qimage1 = QImage(ce1.data, width, height, channel * width, QImage.Format_RGB888)
    
    # height, width, channel = cm2.shape
    # qimage2 = QImage(cm2.data, width, height, channel * width, QImage.Format_RGB888)

    # height, width, channel = ch3.shape
    # qimage3 = QImage(ch3.data, width, height, channel * width, QImage.Format_RGB888)

    # qpixmap1 = QPixmap.fromImage(qimage1)
    # lb1.setPixmap(qpixmap1)

    # qpixmap2 = QPixmap.fromImage(qimage2)
    # lb2.setPixmap(qpixmap2)

    # qpixmap3 = QPixmap.fromImage(qimage3)
    # lb3.setPixmap(qpixmap3)

    # gl.addWidget(lb1, 0, 0)
    # gl.addWidget(lb2, 0, 1)
    # gl.addWidget(lb3, 1, 0)
    # wgt.setLayout(gl)
    # wgt.show()

    # app.exec()


    # fill
    # image = cv2.imread(PATH_IMAGE_HARD_2)

    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # ret, binary_image = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # # 创建一个和原图像大小一样的空白图像
    # mask = np.zeros_like(binary_image)

    # # 填充轮廓
    # cv2.drawContours(mask, contours, -1, (255, 255, 255), thickness=cv2.FILLED)

    # # 将填充后的图像与原图像进行合并
    # result = cv2.bitwise_and(binary_image, mask)

    # cv2.imshow('Filled Image', result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()







