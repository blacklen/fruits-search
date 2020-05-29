# USAGE
# python search.py --index index.csv --query queries/103100.png --result-path dataset

# import the necessary packages
from pyimagesearch.colordescriptor import ColorDescriptor
from pyimagesearch.searcher import Searcher
import argparse
import cv2
import random
import os
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
from hog import hog

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--index", required = True,
	help = "Path to where the computed index will be stored")
ap.add_argument("-q", "--query", required = True,
	help = "Path to the query image")
ap.add_argument("-r", "--result-path", required = True,
	help = "Path to the result path")
args = vars(ap.parse_args())

# initialize the image descriptor
cd = ColorDescriptor((8, 12, 3))

# load the query image and describe it
files = os.listdir(args["query"])
sample = random.sample(files, 1)
for s in sample:
	queryPath = args["query"] + '/' + s
	query = cv2.imread(queryPath)
	# resize image
	# features = hog(queryPath)
	features = cd.histogram2(queryPath)
	# perform the search
	searcher = Searcher(args["index"])
	results = searcher.search(features,1)
	# display the query
	
	cv2.imshow("Query", query)
	cv2.waitKey(1000)
	# searcher.show_histogram(query)

	# loop over the results
	for (score, resultID) in results:
		# load the result image and display it
		result = cv2.imread(args["result_path"] + "/" + resultID)

		cv2.imshow("Result", result)
		cv2.waitKey(1000)
		# searcher.show_histogram(result)
