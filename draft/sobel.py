import numpy as np
import scipy
import scipy.signal as sig
import cv2
import matplotlib.pyplot as plt
from util import resize_image
# With mode="L", we force the image to be parsed in the grayscale, so it is
# actually unnecessary to convert the photo color beforehand.
img_path = "dataset_fruit/Apple/Apple(0).jpg"
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
img = cv2.resize(src=img, dsize=(64, 64))
# img = resize_image(, 64, 128)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Define the Sobel operator kernels.
# kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
# kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

kernel_x = np.array([[-1, 0, 1]])
kernel_y = np.array([[-1], [0], [1]])

G_x = sig.convolve2d(img, kernel_x, mode='same')
G_y = sig.convolve2d(img, kernel_y, mode='same')
# G_x = cv2.filter2D(img, cv2.CV_32F, kernel_x)
# G_y = cv2.filter2D(img, cv2.CV_32F, kernel_y)

# Plot them!
fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

# Actually plt.imshow() can handle the value scale well even if I don't do
# the transformation (G_x + 255) / 2.
ax1.imshow((G_x + 255) / 2, cmap='gray')
ax1.set_xlabel("Gx")
ax2.imshow((G_y + 255) / 2, cmap='gray')
ax2.set_xlabel("Gy")
plt.show()
