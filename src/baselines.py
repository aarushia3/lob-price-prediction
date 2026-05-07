import numpy as np

# make random predictions
def random_baseline(X):
    return np.random.randint(0, 3, size=len(X))

# make predictions based on last training label
def momentum_baseline(X, last_train_label=0):
    preds = np.full(len(X), last_train_label, dtype=int)
    return preds

# make zero predictions for regression
def zero_return_baseline(X):
    return np.zeros(len(X))

# make predictions based on mean training return
def lagged_return_baseline(y_train):
    mean_train_return = np.mean(y_train)
    return mean_train_return  # REMIND this is a scalar btw