import numpy as np
import csv
import matplotlib.pyplot as plt
import cv2

class Searcher:
    def __init__(self, indexPath):
        self.indexPath = indexPath

    def search(self, queryFeatures, limit=4):
        results = {}

        with open(self.indexPath) as f:
            reader = csv.reader(f)

            for row in reader:
                features = [float(x) for x in row[1:]]
                d = self.chi2_distance(features, queryFeatures)
                # d = self.d1(features,queryFeatures)

                results[row[0]] = d
            f.close()

        results = sorted([(v, k) for (k, v) in results.items()])
        return results[:limit]

    def chi2_distance(self, histA, histB, eps=1e-10):
        d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps) for (a, b) in zip(histA, histB)])
        return d

    def d1(self, v1, v2):
        return np.sum(np.absolute(v1 - v2))

	# def show_histogram(self, image):
	# 	for i, col in enumerate(['b', 'g', 'r']):
	# 		hist = cv2.calcHist([image], [i], None, [256], [0, 256])
	# 		plt.plot(hist, color=col)
	# 		plt.xlim([0, 256])
	# 	plt.show()
