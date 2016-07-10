
''' Detect human skin tone and draw a boundary around it.
Useful for gesture recognition and motion tracking.
Usage: 
	python skinDetect.py
	This will start the program. Press any key to exit.
Inspired by: http://stackoverflow.com/a/14756351/1463143
Date: 08 June 2013
Author: Sameer Khan samkhan13.wordpress.com
License: Creative Commons Zero (CC0 1.0) 
https://creativecommons.org/publicdomain/zero/1.0/ 
'''

# Required moduls
import cv2
import numpy
#import Image
from matplotlib import pyplot as plt

def normalize(arr):
    for i in range(3):
        minval = arr[..., i].min()
        maxval = arr[..., i].max()

        if minval != maxval:
            arr[..., i] -= minval
            arr[..., i] *= (255.0 / (maxval - minval)).astype(img.dtype)
    return arr

face_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_eye.xml')

img = cv2.imread('test.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# (image, scale_factor=1.3, min_neighbors=5, flags=0, min_size=(100, 100))
faces = face_cascade.detectMultiScale(gray, 1.3, 5, 0, (100,100))
for (x,y,w,h) in faces:
    # cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]

    # get a smaller region within the roi
    y_temp = (int)(y*1.2)
    x_temp = (int)(x*1.2)
    h_temp = (int)(h/1.2)
    w_temp = (int)(w/1.2)
    roi_gray_inner = gray[y_temp:y+h_temp, x_temp:x+w_temp]
    roi_color_inner = img[y_temp:y+h_temp, x_temp:x+w_temp]
    '''
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    10 12 10 12
    10 8 10 8
    '''

# Constants for finding range of skin color in YCrCb
#min_YCrCb = numpy.array([0,133,77],numpy.uint8)
#max_YCrCb = numpy.array([255,173,127],numpy.uint8)

#img = cv2.imread('kate-upton.jpg')

# Convert image to YCrCb
#imageYCrCb = cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)
imageHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

# split into h,s,v channels
# h, s, v = cv2.split(img)
h_histogram = cv2.calcHist([roi_color_inner],[0], None, [256], [0,256])
s_histogram = cv2.calcHist([roi_color_inner],[1], None, [256], [0,256])

min_HSV = numpy.array([0,75,0], numpy.uint8)
max_HSV = numpy.array([15,190,255], numpy.uint8)
# Find region with skin tone in YCrCb image
skinRegion = cv2.inRange(imageHSV, min_HSV, max_HSV)

# Do contour detection on skin region
contours, hierarchy = cv2.findContours(skinRegion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw the contour on the source image

for i, c in enumerate(contours):
    area = cv2.contourArea(c)
    if area > 1000:
        cv2.drawContours(img, contours, i, (0, 255, 0), 3)

# Display the source image
cv2.imshow('Output',roi_color_inner)
plt.plot(h_histogram, 'H')
plt.xlim([0,256])
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
