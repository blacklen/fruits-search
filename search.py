from descriptor import Descriptor
from searcher import Searcher
import argparse
import cv2
import random
import os
import matplotlib.pyplot as plt
import numpy as np
from fusion import Fusion
from util import insertToFile
from skimage.feature import hog

ap = argparse.ArgumentParser()
index1 = "index-color.csv"
index2 = "index-hog.csv"
index3 = "index-fusion.csv"
resultPath = "dataset_fruit"

ap.add_argument("-q", "--query", required=True)
ap.add_argument("-r", "--descriptor", required=True)
ap.add_argument("-o", "--output", required=True)

args = vars(ap.parse_args())

descriptor = Descriptor()
files = os.listdir(args["query"])
sample = random.sample(files, 10)

for s in sample:
    queryPath = args["query"] + '/' + s
    query = cv2.imread(queryPath)

    content = "Query : %s\n" % (queryPath)
    insertToFile(args["output"], content)

    cv2.imshow("Query", query)
    cv2.waitKey(1000)
    index = index1
    if args["descriptor"] == 'color':
        features = descriptor.color(queryPath)
        index = index1
    elif args["descriptor"] == 'hog':
        features = descriptor.hog(queryPath)
        index = index2
    else:
        features = descriptor.fusion(queryPath)
        index = index3

    searcher = Searcher(index)
    results = searcher.search(features, args["output"], 1)
    for (score, resultID) in results:
        result = cv2.imread(resultPath + "/" + resultID)
        cv2.imshow("Result", result)
        cv2.waitKey(1000)
