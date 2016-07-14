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
    h = image[:,:,0]
    s = image[:,:,1]
    #v = image[:,:,2]

    h_mean = calculateMean(h)
    h_sd = calculateSD(h, h_mean)
    s_mean = calculateMean(s)
    s_sd = calculateSD(s, s_mean)
    #v_mean = calculateMean(c)
    #v_sd = calculateSD(c, c_mean)
    lowerBound = cv.Scalar(h_mean - h_sd*2, s_mean - s_sd*2, 0)
    upperBound = cv.Scalar(h_mean + h_sd*2, s_mean + s_sd*2, 255)
    # 1 1 1
    # 2 1 1
    return lowerBound, upperBound



img = cv2.imread('test.jpg')
#img = cv2.imread('test2.jpg')
#img = cv2.imread('kate-upton.jpg')

gray_conv = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray, color, shape = faceDetection(img, gray_conv)

img_tmp = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
color_tmp = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)

lower, upper = calculateUpperAndLowerBounds(color_tmp)

x, y, w, h = shape
cv2.rectangle(img_tmp, (x, int(y * 0.5)), (x + w, y + int(h*1.5)), (0, 0, 0), -1)

mask = cv2.inRange(img_tmp, lower, upper)
skinRegion = cv2.bitwise_and(img_tmp, img_tmp, mask=mask)



# Do contour detection on skin region
contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw the contour on the source image
for i, c in enumerate(contours):
    area = cv2.contourArea(c)
    if area > 1000:
        cv2.drawContours(img, contours, i, (0, 255, 0), 3)


cv2.imshow('test', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
