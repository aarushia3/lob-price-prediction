from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# ridge regression with all features
def train_ridge(X_train, y_train, alpha=1.0):
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    return model

# check if pred sign matches the true sign
def directional_accuracy(preds, y_test):
    mask = y_test != 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.sign(preds[mask]) == np.sign(y_test[mask])))

# evaluate regression model by computing MAE, MSE, RMSE, dir accuracy
def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    mae  = mean_absolute_error(y_test, preds)
    mse  = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    dir_acc = directional_accuracy(preds, y_test)
    return {"mae": mae, "mse": mse, "rmse": rmse, "dir_acc": dir_acc}