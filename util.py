import numpy as np

def insertToFile(path, contents):
    file_object = open(path, 'a')
    file_object.write(contents)
    file_object.close()

def normalize(v):
    max_value = max(v)
    min_value = min(v)
    return np.array([2 * (v[i] - min_value) / (max_value - min_value) - 1 for i in range(len(v))])

