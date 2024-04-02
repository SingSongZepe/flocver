import cv2
import numpy

from value.strings import IMAGE_SHOW_TITLE

def show_single_image(binary_image: numpy.array, title=IMAGE_SHOW_TITLE) -> None:
    cv2.imshow(title, binary_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

