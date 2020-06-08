# import the necessary packages
import numpy as np
import cv2
import imutils
import matplotlib.pyplot as plt
import itertools
from skimage.feature import hog
from skimage import color
from numpy import linalg as LA

n_slice = 6 # slice image
n_bin = 10
n_orient = 8
p_p_c = (2, 2)
c_p_b = (1, 1)
h_type = 'region'  # global or region
d_type = 'd1'      # distance type

bins = (8,12,3)
depth = 5  # retrieved depth, set to None will count the ap for whole database

class Descriptor:
    
    def color(self, imagePath):
        img = cv2.imread(imagePath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([img], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        for i in range(1, len(hist)):
            hist[i] += hist[i - 1]
        return hist


    def _HOG(self, img, n_bin, normalize=True):
        image = color.rgb2gray(img)
        fd = hog(image, orientations=n_orient, pixels_per_cell=p_p_c, cells_per_block=c_p_b)
        bins = np.linspace(0, np.max(fd), n_bin+1, endpoint=True)
        hist, _ = np.histogram(fd, bins=bins)
    
        if normalize:
            hist = np.array(hist) / np.sum(hist)
    
        return hist

    def histogram(self, input, n_bin=n_bin, type=h_type, n_slice=n_slice, normalize=True):
        img = cv2.imread(input)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, channel = img.shape
        img = cv2.resize(src=img, dsize=(64, 128))

        if type == 'global':
            hist = self._HOG(img, n_bin)
                
        elif type == 'region':
            hist = np.zeros((n_slice, n_slice, n_bin))
            h_silce = np.around(np.linspace(0, height, n_slice+1, endpoint=True)).astype(int)
            w_slice = np.around(np.linspace(0, width, n_slice + 1, endpoint=True)).astype(int)
        
        for hs in range(len(h_silce)-1):
            for ws in range(len(w_slice)-1):
                img_r = img[h_silce[hs]:h_silce[hs+1], w_slice[ws]:w_slice[ws+1]]  # slice img to regions
                hist[hs][ws] = self._HOG(img_r, n_bin)
                
        if normalize:
            hist /= np.sum(hist)
                
        return hist.flatten()

    def hog(self,img_path, cell_size=8, block_size=2, bins=9):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(src=img, dsize=(64, 128))
        h, w = img.shape  # 128, 64
        
        # gradient
        xkernel = np.array([[-1, 0, 1]])
        ykernel = np.array([[-1], [0], [1]])
        dx = cv2.filter2D(img, cv2.CV_32F, xkernel)
        dy = cv2.filter2D(img, cv2.CV_32F, ykernel)
        
        # histogram
        magnitude = np.sqrt(np.square(dx) + np.square(dy))
        orientation = np.arctan(np.divide(dy, dx+0.00001)) # radian
        orientation = np.degrees(orientation) # -90 -> 90
        orientation += 90 # 0 -> 180
        
        num_cell_x = w // cell_size # 8
        num_cell_y = h // cell_size # 16
        hist_tensor = np.zeros([num_cell_y, num_cell_x, bins]) # 16 x 8 x 9
        for cx in range(num_cell_x):
            for cy in range(num_cell_y):
                ori = orientation[cy*cell_size:cy*cell_size+cell_size, cx*cell_size:cx*cell_size+cell_size]
                mag = magnitude[cy*cell_size:cy*cell_size+cell_size, cx*cell_size:cx*cell_size+cell_size]
                # https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram.html
                hist, _ = np.histogram(ori, bins=bins, range=(0, 180), weights=mag) # 1-D vector, 9 elements
                hist_tensor[cy, cx, :] = hist
            pass
        pass
        
        # normalization
        redundant_cell = block_size-1
        feature_tensor = np.zeros([num_cell_y-redundant_cell, num_cell_x-redundant_cell, block_size*block_size*bins])
        for bx in range(num_cell_x-redundant_cell): # 7
            for by in range(num_cell_y-redundant_cell): # 15
                by_from = by
                by_to = by+block_size
                bx_from = bx
                bx_to = bx+block_size
                v = hist_tensor[by_from:by_to, bx_from:bx_to, :].flatten() # to 1-D array (vector)
                feature_tensor[by, bx,:] = v / LA.norm(v, 2)
                # avoid NaN:
                if np.isnan(feature_tensor[by, bx, :]).any(): # avoid NaN (zero division)
                    feature_tensor[by, bx, :] = v
        
        return feature_tensor.flatten()  # 3780 features
        
    def fusion(self, imagePath):
        color = self.color(imagePath)
        hog = self.hog(imagePath)
        hist = np.concatenate((color, hog), axis = 0)
        return self.normalize(hist)
        
    def normalize(self,v):
        max_value = max(v)
        min_value = min(v)
        return [(v[i] - min_value) / (max_value - min_value) for i in range(len(v))]
        
	
