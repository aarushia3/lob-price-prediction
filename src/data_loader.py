import numpy as np

def load_fi2010(path):
    data = np.loadtxt(path)

    if data.shape[0] < data.shape[1]:
        data = data.T
    
    X = data[:, :-5]
    y = data[:, -5].flatten()
    return X, y