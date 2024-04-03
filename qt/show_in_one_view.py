import cv2
import numpy as np

from PySide6.QtWidgets import QApplication, QLabel, QGridLayout, QWidget
from PySide6.QtGui import QPixmap, QImage

from value.value import *

def show_in_one_view(images: np.array) -> None:
    app = QApplication([])
    wgt = QWidget()
    gl = QGridLayout()

    for idx, image in enumerate(images):
        height, width, channel = image.shape
        qimage = QImage(image.data, width, height, channel * width, QImage.Format_RGB888)
        qpixmap = QPixmap.fromImage(qimage)
        lb = QLabel()
        lb.setPixmap(qpixmap)
        
        ridx = idx // MAX_ROW_LIMIT;
        cidx = idx % MAX_ROW_LIMIT;
        gl.addWidget(lb, ridx, cidx)

    wgt.setLayout(gl)
    wgt.show()
    app.exec()