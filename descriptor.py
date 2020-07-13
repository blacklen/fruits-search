import numpy as np
import cv2
import imutils
import matplotlib.pyplot as plt
import itertools
from skimage.feature import hog
from skimage import color
from util import normalize, resize_image
import scipy.signal as sig

class Descriptor:

    def __init__(self):
        pass

    def fusion(self, imagePath):
        img = resize_image(imagePath, 64, 128)
        color = self.color(img)
        hog = self.hog(img)
        hist = np.concatenate((color, hog), axis=None)
        return hist
    
    def color(self, input):
        if isinstance(input, np.ndarray):  # examinate input type
            img = input.copy()
        else:
            img = cv2.imread(input)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([img], [0, 1, 2], None, (8,12,3), [0, 180, 0, 256, 0, 256])
        hist = hist.flatten()
        for i in range(1, len(hist)):
            hist[i] += hist[i - 1]
        return normalize(hist)

    def hog(self, input, cell_size=8, block_size=2, bins=9):
        if isinstance(input, np.ndarray):  # examinate input type
            img = input.copy()
        else:
            img = resize_image(input, 64, 128)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = img.shape
        
        xkernel = np.array([[-1, 0, 1]])
        ykernel = np.array([[-1], [0], [1]])

        dx = sig.convolve2d(img, xkernel, mode='same')
        dy = sig.convolve2d(img, ykernel, mode='same')
        
        magnitude = np.sqrt(np.square(dx) + np.square(dy))
        orientation = np.arctan(np.divide(dy, dx))
        orientation = np.degrees(orientation)
        orientation += 90

        num_cell_x = w // cell_size
        num_cell_y = h // cell_size
        hist = np.zeros([num_cell_y, num_cell_x, bins])
        for cx in range(num_cell_x):
            for cy in range(num_cell_y):
                directions = orientation[cy*cell_size:cy*cell_size + cell_size, cx*cell_size:cx*cell_size+cell_size]
                mag = magnitude[cy*cell_size:cy*cell_size+cell_size, cx*cell_size:cx*cell_size+cell_size]
                bucket_vals = np.zeros(bins)
                for (m, d) in zip(mag.flatten(), directions.flatten()):
                    self.assign_bucket_vals(m, d, bucket_vals, bins)
                hist[cy, cx, :] = bucket_vals
        
        # normalization
        redundant_cell = block_size-1
        features = np.zeros([num_cell_y-redundant_cell, num_cell_x-redundant_cell, block_size*block_size*bins])
        for bx in range(num_cell_x - redundant_cell):
            for by in range(num_cell_y - redundant_cell):
                by_from = by
                by_to = by+block_size
                bx_from = bx
                bx_to = bx+block_size
                v = hist[by_from:by_to, bx_from:bx_to, :].flatten()
                features[by, bx, :] = v / np.linalg.norm(v)

                if np.isnan(features[by, bx, :]).any():
                    features[by, bx, :] = v
        
        return normalize(features.flatten())
    
    def assign_bucket_vals(self, m, d, bucket_vals, bins):
        if not np.isnan(d):
            # Handle the case when the direction is between [160, 180)
            if d >= 160:
                left_bin = bins - 1
                right_bin = 0
                left_val = m * (bins * 20 - d) / 20
            else:
                left_bin = int(d / 20.)
                right_bin = (int(d / 20.) + 1) % bins
                left_val = m * (right_bin * 20 - d) / 20
            
            right_val=m * (d - left_bin * 20) / 20

            bucket_vals[left_bin] += left_val
            bucket_vals[right_bin] += right_val

    
                
    
	
