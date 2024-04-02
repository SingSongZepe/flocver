import cv2
import numpy as np

from PySide6.QtWidgets import QApplication, QLabel, QGridLayout, QWidget
from PySide6.QtGui import QPixmap, QImage

def show_in_one_view(images: np.array) -> None:
    app = QApplication([])
    wgt = QWidget()
    gl = QGridLayout()

    for image in images:
        pass
        
        # gl.addWidget()

    wgt.setLayout(gl)
    wgt.show()

    app.exec()