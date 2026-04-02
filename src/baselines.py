import numpy as np

def random_baseline(X):
    return np.random.randint(0, 3, size=len(X))

def momentum_baseline(X, last_train_label=0):
    preds = np.full(len(X), last_train_label, dtype=int)
    return preds