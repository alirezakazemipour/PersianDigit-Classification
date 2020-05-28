import numpy as np
import cv2
from skimage import feature

fname = '01.jpg'

numPoints = 28
radius = 3

image = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
lbp = feature.local_binary_pattern(image, numPoints, radius)
J = np.copy(lbp)
J = np.array(J, dtype=np.float32)
cv2.imshow('J', J)
cv2.waitKey(0)
