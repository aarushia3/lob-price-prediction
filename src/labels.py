import numpy as np

def process_labels(y):
    y = np.array(y).flatten().astype(int)
    remap = {1: 2, 2: 1, 3: 0}
    return np.array([remap[v] for v in y])