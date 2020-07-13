import numpy as np
import csv
import matplotlib.pyplot as plt
import cv2
from util import insertToFile

class Searcher:
    def __init__(self, indexPath):
        self.indexPath = indexPath

    def search(self, queryFeatures, outputPath, limit=4):
        results = {}

        with open(self.indexPath) as f:
            reader = csv.reader(f)

            for row in reader:
                features = [float(x) for x in row[1:]]
                d = self.euclidean(features,queryFeatures)

                results[row[0]] = d
            f.close()

        results = sorted([(v, k) for (k, v) in results.items()])
        result = [str(k) for (v, k) in results[:limit]]
        insertToFile(outputPath,"Result: %s\n\n" % ( ",".join(result)))
        return results[:limit]

    def euclidean(self, v1, v2):
        return np.sqrt(np.sum(np.square(v1 - v2)))
        
    
