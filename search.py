from descriptor import Descriptor
from searcher import Searcher
import argparse
import cv2
import random
import os
import matplotlib.pyplot as plt
import numpy as np
from util import insertToFile
from skimage.feature import hog

ap = argparse.ArgumentParser()
resultPath = "dataset_fruit"
queryset = "query-fruit"

ap.add_argument("-r", "--descriptor", required=True)
ap.add_argument("-o", "--output", required=True)
ap.add_argument("-i", "--index", required=True)

args = vars(ap.parse_args())

descriptor = Descriptor()
files = os.listdir(queryset)
sample = random.sample(files, 10)

for s in sample:
    queryPath = queryset + '/' + s
    query = cv2.imread(queryPath)

    content = "Query : %s\n" % (queryPath)
    insertToFile(args["output"], content)

    cv2.imshow("Query", query)
    cv2.waitKey(1000)

    if args["descriptor"] == 'color':
        features = descriptor.color(queryPath)
    elif args["descriptor"] == 'hog':
        features = descriptor.hog(queryPath)
    else:
        features = descriptor.fusion(queryPath)

    searcher = Searcher(args["index"])
    results = searcher.search(features, args["output"], 1)
    for (score, resultID) in results:
        result = cv2.imread(resultPath + "/" + resultID)
        cv2.imshow("Result", result)
        cv2.waitKey(1000)
