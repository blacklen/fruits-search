from descriptor import Descriptor
import argparse
import glob
import cv2
import os

ap = argparse.ArgumentParser()
dataset = "dataset_fruit"

ap.add_argument("-r", "--index", required=True)
ap.add_argument("-d", "--descriptor", required=True)
args = vars(ap.parse_args())

descriptor = Descriptor()

output = open(args["index"], "w")
files = os.listdir(dataset)
for file in files:
    for imagePath in glob.glob(dataset + "/" + file + "/*.jpg"):

        imageID = imagePath[imagePath.rfind("/") + 1:]
        if args['descriptor'] == 'color':
            features = descriptor.color(imagePath)
        elif args['descriptor'] == 'hog':
            features = descriptor.hog(imagePath)
        else:
            features = descriptor.fusion(imagePath)

        features = [str(f) for f in features]
        output.write("%s,%s\n" % (imageID, ",".join(features)))

output.close()
