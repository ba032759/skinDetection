# Required moduls
import cv
import cv2
from math import sqrt


def calculateMean(channel_in):
    channel = channel_in.copy()
    height, width = channel.shape
    counter = 0
    total = 0
    for i in range(0, height):
        for j in range(0, width):
            if channel[i, j] > 0:
                total += channel[i, j]
                counter += 1
    return total / counter


def calculateSD(channel_in, mean):
    channel = channel_in.copy()
    height, width = channel.shape
    counter = 0
    variance = 0
    for i in range(0, height):
        for j in range(0, width):
            if channel[i, j] > 0:
                variance += (float(channel[i, j]) - mean) ** 2
                counter += 1
    variance = variance / counter
    return sqrt(variance)


def faceDetection(img, gray):
    face_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_default.xml')
    # eye_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_eye.xml')

    # (image, scale_factor=1.3, min_neighbors=5, flags=0, min_size=(100, 100))
    faces = face_cascade.detectMultiScale(gray, 1.3, 5, 0, (100, 100))
    for (x, y, w, h) in faces:
        roi = img[y:y + h, x:x + w]
        # get a smaller region within the roi
        y_temp = int(y * 1.2)
        x_temp = int(x * 1.2)
        h_temp = int(h / 1.2)
        w_temp = int(w / 1.2)
        roi_inner_gray = gray[y_temp:y + h_temp, x_temp:x + w_temp]
        roi_inner = img[y_temp:y + h_temp, x_temp:x + w_temp]
        return roi_inner_gray, roi_inner, (x, y, w, h)
        '''
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        10 12 10 12
        10 8 10 8
        '''
def calculateUpperAndLowerBounds(image):
    a = image[:,:,0]
    b = image[:,:,1]
    c = image[:,:,2]

    a_mean = calculateMean(a)
    a_sd = calculateSD(a, a_mean)
    b_mean = calculateMean(b)
    b_sd = calculateSD(b, b_mean)
    c_mean = calculateMean(c)
    c_sd = calculateSD(c, c_mean)
    lowerBound = cv.Scalar(0, b_mean - b_sd*2, c_mean - c_sd*2)
    upperBound = cv.Scalar(a_mean + a_sd*2, b_mean + b_sd*2, c_mean + c_sd*2)
    # 1 1 1
    # 2 1 1
    return lowerBound, upperBound



img = cv2.imread('test.jpg')
#img = cv2.imread('test2.jpg')
#img = cv2.imread('kate-upton.jpg')

gray_conv = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray, color, shape = faceDetection(img, gray_conv)

img_tmp = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
color_tmp = cv2.cvtColor(color, cv2.COLOR_BGR2YCR_CB)

lower, upper = calculateUpperAndLowerBounds(color_tmp)

x, y, w, h = shape
cv2.rectangle(img_tmp, (x, int(y * 0.5)), (x + w, y + int(h*1.5)), (0, 0, 0), -1)

mask = cv2.inRange(img_tmp, lower, upper)
skinRegion = cv2.bitwise_and(img_tmp, img_tmp, mask=mask)


# Do contour detection on skin region
contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw largest contour on the source image
largest_area = 0.0
largest_area_index = 0
for i, c in enumerate(contours):
    area = cv2.contourArea(c)
    if area > largest_area:
        largest_area = area
        largest_area_index = i
        bounding_rect = cv2.boundingRect(contours[largest_area_index])
        largest_contour = contours[largest_area_index]

#cv2.drawContours(img, contours, largest_area_index, (0, 255, 0), 3)
x, y, w, h = bounding_rect
scale_factor = 1.3 #1.4
#cv2.rectangle(img, (int(scale_factor*x), int(scale_factor*y)), (x + int(w/scale_factor), y + int(h/scale_factor)), (0, 255, 0), 2)

color = img[int(y/scale_factor):y + h, x:x + w]
'''
scale_factor = 1 #1.4
color = img[int(scale_factor*y):y + int(h/scale_factor), int(scale_factor*x):x + int(w/scale_factor)]

# find skinColor from hand
color_tmp = cv2.cvtColor(color, cv2.COLOR_BGR2YCR_CB)

lower, upper = calculateUpperAndLowerBounds(color_tmp)

mask = cv2.inRange(img_tmp, lower, upper)
skinRegion = cv2.bitwise_and(img_tmp, img_tmp, mask=mask)
contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw largest contour on the source image
largest_area = 0.0
largest_area_index = 0
for i, c in enumerate(contours):
    area = cv2.contourArea(c)
    if area > largest_area:
        largest_area = area
        largest_area_index = i
        bounding_rect = cv2.boundingRect(contours[largest_area_index])
        largest_contour = contours[largest_area_index]
cv2.drawContours(img, contours, largest_area_index, (0, 255, 0), 3)
(x,y),radius = cv2.minEnclosingCircle(largest_contour)
center = (int(x),int(y))
radius = int(radius/2)
cv2.circle(img,center,radius,(0,255,0),2)
'''
cv2.imshow('test', color)
cv2.waitKey(0)
cv2.destroyAllWindows()
