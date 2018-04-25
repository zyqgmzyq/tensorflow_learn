import cv2 as cv
import numpy as np


def create():
    img = np.ones([400, 400, 1], dtype=int)
    # img = np.array([[1.7, 3.2], [34.2, 4.8]], dtype=float)
    img = img * 127
    cv.imshow("image", img)


create()
cv.waitKey(0)
cv.destroyAllWindows()

