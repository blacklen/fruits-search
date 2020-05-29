# import the necessary packages
import numpy as np
import cv2
import imutils
import matplotlib.pyplot as plt
import itertools
from hog import hog

n_bin   = 12        # histogram bins
n_slice = 3         # slice image
h_type  = 'region'  # global or region
d_type  = 'd1'      # distance type

depth = 3  # retrieved depth, set to None will count the ap for whole database

class ColorDescriptor:
	def __init__(self, bins):
		# store the number of bins for the 3D histogram
		self.bins = bins

	def _count_hist(self, input, n_bin, bins, channel):
		img = input.copy()
		print("bin", np.arange(n_bin))
		bins_idx = {key: idx for idx, key in enumerate(itertools.product(np.arange(n_bin), repeat=channel))}  # permutation of bins
		hist = np.zeros(n_bin ** channel)
		for idx in range(len(bins)-1):
    			# print(img[(input >= bins[idx]) & (input < bins[idx+1])])
			img[(input >= bins[idx]) & (input < bins[idx+1])] = idx
		# add pixels into bins
		height, width, _ = img.shape
		for h in range(height):
			for w in range(width):
				b_idx = bins_idx[tuple(img[h, w])]
				hist[b_idx] += 1
			
		return hist
	
	def histogram2(self, input, n_bin=n_bin, type=h_type, n_slice=n_slice, normalize=True):
		img = cv2.imread(input, cv2.COLOR_BGR2RGB)
		height, width, channel = img.shape
		bins = np.linspace(0, 256, n_bin + 1, endpoint=True)  # slice bins equally for each channel

		if type == 'global':
			hist = self._count_hist(img, n_bin, bins, channel)
		
		elif type == 'region':
			hist = np.zeros((n_slice, n_slice, n_bin ** channel))
			h_silce = np.around(np.linspace(0, height, n_slice+1, endpoint=True)).astype(int)
			w_slice = np.around(np.linspace(0, width, n_slice + 1, endpoint=True)).astype(int)
		
		for hs in range(len(h_silce)-1):
			for ws in range(len(w_slice)-1):
				img_r = img[h_silce[hs]:h_silce[hs+1], w_slice[ws]:w_slice[ws+1]]  # slice img to regions
				hist[hs][ws] = self._count_hist(img_r, n_bin, bins, channel)
		
		if normalize:
			hist /= np.sum(hist)
		
		return hist.flatten()	

	def describe(self, image):
		# convert the image to the HSV color space and initialize
		# the features used to quantify the image
		# b, g, r = cv2.split(image)
		# red = cv2.equalizeHist(r)
		# green = cv2.equalizeHist(g)
		# blue = cv2.equalizeHist(b)
		# image = cv2.merge((blue, green, red))
		image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
		print(image)

		# Histogram equalisation on the V-channel
		image[:, :, 2] = cv2.equalizeHist(image[:, :, 2])
		features = []

		# grab the dimensions and compute the center of the image
		(h, w) = image.shape[:2]
		(cX, cY) = (int(w * 0.5), int(h * 0.5))

		# divide the image into four rectangles/segments (top-left,
		# top-right, bottom-right, bottom-left)
		segments = [(0, cX, 0, cY), (cX, w, 0, cY), (cX, w, cY, h),
			(0, cX, cY, h)]

		# construct an elliptical mask representing the center of the
		# image
		(axesX, axesY) = (int(w * 0.75) // 2,  int(h * 0.75) // 2)
		ellipMask = np.zeros(image.shape[:2], dtype = "uint8")
		cv2.ellipse(ellipMask, (cX, cY), (axesX, axesY), 0, 0, 360, 255, -1)

		# loop over the segments
		# for (startX, endX, startY, endY) in segments:
		# 	# construct a mask for each corner of the image, subtracting
		# 	# the elliptical center from it
		# 	cornerMask = np.zeros(image.shape[:2], dtype = "uint8")
		# 	cv2.rectangle(cornerMask, (startX, startY), (endX, endY), 255, -1)
		# 	cornerMask = cv2.subtract(cornerMask, ellipMask)

		# 	# extract a color histogram from the image, then update the
		# 	# feature vector
		# 	hist = self.histogram(image, cornerMask)
		# 	features.extend(hist)

		# extract a color histogram from the elliptical region and
		# update the feature vector
		# hist = self.histogram(image, ellipMask)
		# features.extend(hist)
		hist = self.histogram(image, None)
		features.extend(hist) 

		# return the feature vector
		return features

	def histogram(self, image, mask):
		# extract a 3D color histogram from the masked region of the
		# image, using the supplied number of bins per channel
		hist = cv2.calcHist([image], [0, 1, 2], mask, self.bins, [0, 180, 0, 256, 0, 256])
		
		hist = cv2.normalize(hist, hist).flatten()
		for i in range(1,len(hist)):
    			hist[i] += hist[i - 1]
		# return the histogram
		return hist
		
