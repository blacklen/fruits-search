import cv2
import numpy as np

image = cv2.imread("fruits/114_100.jpg")

# convert image from RGB to HSV
img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

# Histogram equalisation on the V-channel
img_hsv[:, :, 0] = cv2.equalizeHist(img_hsv[:, :, 0])

# convert image back from HSV to RGB
image = cv2.cvtColor(img_hsv, cv2.COLOR_YUV2BGR)

cv2.imshow("equalizeHist", image)
cv2.waitKey(1000)