import cv
import cv2
import numpy as np
from matplotlib import pyplot as plt
from math import sqrt

#img = cv2.imread('test.jpg')
img = cv2.imread('kate-upton.jpg')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

hsv = cv2.inRange(hsv, np.array([0, 0, 0],np.uint8), np.array([179, 255, 255],np.uint8))
h, s, v = cv2.split(hsv)

cv2.imshow('Output',hsv)
cv2.waitKey(0)
cv2.destroyAllWindows()
