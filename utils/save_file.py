import cv2
import numpy as np

def save_file(image: np.array, path: str) -> bool:
    cv2.imwrite(path, image)
