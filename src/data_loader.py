import numpy as np

def load_fi2010(path):
    data = np.loadtxt(path)

    if data.shape[0] < data.shape[1]:
        data = data.T
    
    X = data[:, :144]
    y = data[:, -1] # row 149 is k = 10 horizon
    y = y.flatten()
    return X, y

def load_fi2010_regression(path, horizon=10):
    data = np.loadtxt(path)

    if data.shape[0] < data.shape[1]:
        data = data.T

    X = data[:, :144]

    best_ask = data[:, 0]
    best_bid = data[:, 2]
    mid_price = (best_ask + best_bid) / 2.0

    n = len(mid_price)
    returns = mid_price[horizon:] / mid_price[:n - horizon] - 1.0

    # dont need last horizon rows of X since we dont have return for thosw
    X_reg = X[:n - horizon]

    return X_reg, returns