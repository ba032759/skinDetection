import numpy as np
import cv2
import Image
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
cv2.imshow('Face region',roi_gray_inner)
'''
hist,bins = np.histogram(roi_gray_inner.flatten(),256,[0,256])

cdf = hist.cumsum()
cdf_normalized = cdf * hist.max()/ cdf.max()

plt.plot(cdf_normalized, color = 'b')
plt.hist(roi_gray_inner.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')
plt.show()
'''
#img = Image.open('orig.jpg').convert('RGBA')
test = Image.open('test.jpg').convert('RGBA')
a = np.array(roi_gray_inner)
b = normalize(a)

im = Image.fromarray(b)
cv2.imshow('test', b)
cv2.waitKey(0)
cv2.destroyAllWindows()

