
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
from matplotlib import pyplot as plt
from math import sqrt

def calculateMean(channel_in):
    channel = channel_in.copy()
    height, width = channel.shape
    counter = 0
    total = 0
    for i in range(0, height):
        for j in range(0, width):
            if channel[i,j] > 0:
                total += channel[i,j]
                counter += 1;
    return total/counter

def calculateSD(channel_in, mean):
    channel = channel_in.copy()
    height, width = channel.shape
    counter = 0
    variance = 0
    for i in range(0, height):
        for j in range(0, width):
            if channel[i,j] > 0:
                variance += (float(channel[i,j]) - mean)**2
                counter += 1
    variance = variance/(counter)
    return sqrt(variance)

def removeExtremeValues(channel_in, lowerThreshold, upperThreshold):
    channel = channel_in.copy()
    height, width = channel.shape
    for i in range(0, height):
        for j in range(0, width):
            if channel[i,j] < lowerThreshold or channel[i,j] > upperThreshold:
                channel[i,j] = 0
    return channel

# base means the image with the zero values to remove from the image
def removeZeroValues(base, image_in):
    height, width = base.shape
    image = image_in.copy()
    for i in range(0, height):
        for j in range(0, width):
            if base[i,j] == 0:
                image[i,j,0] = 0
                image[i,j,1] = 0
                image[i,j,2] = 0
    return image

# normalize R and G and keep R
def convertTo_rgR(image):
    height, width = base.shape
    new_image = image.copy()
    new_image = new_image.astype(float)
    for i in range(0, height):
        for j in range(0, width):
            if (int(image[i,j,0]) + int(image[i,j,1]) + int(image[i,j,2])) == 0:
                new_image[i,j,0] = 0
                new_image[i,j,1] = 0
            else:
                new_image[i,j,0] = float(image[i,j,2]) / (int(image[i,j,0]) + int(image[i,j,1]) + int(image[i,j,2]))
                new_image[i,j,1] = float(image[i,j,1]) / (int(image[i,j,0]) + int(image[i,j,1]) + int(image[i,j,2]))
    return new_image


face_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_eye.xml')

#img = cv2.imread('test.jpg')
img = cv2.imread('kate-upton.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# (image, scale_factor=1.3, min_neighbors=5, flags=0, min_size=(100, 100))
faces = face_cascade.detectMultiScale(gray, 1.3, 5, 0, (100,100))
for (x,y,w,h) in faces:
    # cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi = img[y:y+h, x:x+w]

    # get a smaller region within the roi
    y_temp = (int)(y*1.2)
    x_temp = (int)(x*1.2)
    h_temp = (int)(h/1.2)
    w_temp = (int)(w/1.2)
    roi_inner_gray = gray[y_temp:y+h_temp, x_temp:x+w_temp]
    roi_inner = img[y_temp:y+h_temp, x_temp:x+w_temp]
    '''
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    10 12 10 12
    10 8 10 8
    '''

# remove blue values
b, g, r = cv2.split(roi_inner)
mean_b = calculateMean(b)
sd_b = calculateSD(b, mean_b)
img2 = cv2.inRange(img, numpy.array([mean_b-sd_b*2, 0, 0]), numpy.array([mean_b+sd_b*2, 255, 255]))
cv2.imshow('Output',img2)

mean = calculateMean(roi_inner_gray)
sd = calculateSD(roi_inner_gray, mean)

base = removeExtremeValues(roi_inner_gray, mean - sd*2, mean + 3*sd)

clean_inner = removeZeroValues(base, roi_inner)


test = convertTo_rgR(clean_inner)
mean_r = calculateMean(test[:,:,0])
mean_g = calculateMean(test[:,:,1])
mean_R = calculateMean(test[:,:,2])
print("rgR mean values")
print(mean_r)
print(mean_g)
print(mean_R)

sd_r = calculateSD(test[:,:,0], mean_r)
sd_g = calculateSD(test[:,:,1], mean_g)
sd_R = calculateSD(test[:,:,2], mean_R)
print("rgR sd values")
print(sd_r)
print(sd_g)
print(sd_R)

img_rgR = convertTo_rgR(img)
img_r = removeExtremeValues(img_rgR[:,:,0], mean_r - sd_r/2, mean_r + sd_r/2)
img_g = removeExtremeValues(img_rgR[:,:,1], mean_g - sd_g/2, mean_g + sd_g/2)
img_R = removeExtremeValues(img_rgR[:,:,2], mean_R - sd_R/2, mean_R + sd_R/2)
#testout = cv2.merge([img_r, img_g, img_R])
#cv2.imshow("hope", testout)

'''
# Find region with skin tone in YCrCb image
skinRegion = cv2.inRange(img_rgR,
                         numpy.array([mean_r - sd_r/2, mean_g - sd_g, mean_R + sd_R]),
                         numpy.array([mean_r + sd_r/2, mean_g + sd_g, mean_R + sd_R]))
# Do contour detection on skin region
contours, hierarchy = cv2.findContours(skinRegion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw the contour on the source image
for i, c in enumerate(contours):
    area = cv2.contourArea(c)
    if area > 1000:
        cv2.drawContours(img, contours, i, (0, 255, 0), 3)
'''
# Display the source image
#cv2.imshow('Output',skinRegion)
#plt.plot(gray_histogram, 'G')
#plt.xlim([0,256])
#plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
