# USAGE
# python index.py --dataset dataset --index index.csv

# import the necessary packages
from pyimagesearch.colordescriptor import ColorDescriptor
import argparse
import glob
import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required = True,
	help = "Path to the directory that contains the images to be indexed")
ap.add_argument("-i", "--index", required = True,
	help = "Path to where the computed index will be stored")
args = vars(ap.parse_args())

# initialize the color descriptor
cd = ColorDescriptor((8, 12, 3))

# open the output index file for writing
output = open(args["index"], "w")
files = os.listdir(args["dataset"])
for file in files:
	for imagePath in glob.glob(args["dataset"] + "/" + file + "/*.jpg"):

		imageID = file + "/" + imagePath[imagePath.rfind("/") + 1 :]

		image = cv2.imread(imagePath)
		features = cd.describe(image)

		features = [str(f) for f in features]
		output.write("%s,%s\n" % (imageID, ",".join(features)))

# close the index file
output.close()