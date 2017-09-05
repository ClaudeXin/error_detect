import numpy as np


def normalize(X):
    # max_array = np.matrix([X.argmax(0)])
    max_array = X.argmax(0)
    return np.apply_along_axis(lambda x: x / (1 + max_array), 1, X)

