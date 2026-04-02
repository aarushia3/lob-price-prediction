import numpy as np

def load_fi2010(path):
    data = np.loadtxt(path)

    if data.shape[0] < data.shape[1]:
        data = data.T
    
    X = data[:, :144]
    y = data[:, -1]     # row 149 = k=10 horizon
    y = y.flatten()
    return X, y