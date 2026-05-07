# mip based feature selection for regression
# given (X, y), n samples, p features, select atmost k features to minimize MSE of a linear predictor
# variables: w \in R^p, z \in {0,1)^p, b \in R}
# objective: min (1/n) ||y - Xw - b||^2
# constraints: sum(z) <= k, -M * z_j <= w_j <= M * z_j for all j (trying big-M linking)

import numpy as np
import gurobipy as gp
from gurobipy import GRB
from sklearn.metrics import mean_absolute_error, mean_squared_error

# solve MIP feature selection problem, return selected features, weights, intercept, and final mip gap
def select_features_mip(X_train, y_train, k, M=None, time_limit=60,
                        max_samples=5000, seed=42):
    from sklearn.linear_model import Ridge

    # subsample to make dataset smaller bec this was literally taking forever
    n_full = len(X_train)
    if n_full > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n_full, size=max_samples, replace=False)
        X_sub = X_train[idx]
        y_sub = y_train[idx]
    else:
        X_sub, y_sub = X_train, y_train

    n, p = X_sub.shape

    # find M by fitting Ridge and scaling up the max coef
    if M is None:
        ridge = Ridge(alpha=1.0).fit(X_sub, y_sub)
        M = float(np.abs(ridge.coef_).max() * 20)
        M = max(M, 1e-4)

    # build normal equation matrices for objective
    ones  = np.ones((n, 1))
    X_aug = np.hstack([X_sub, ones])
    y_vec = y_sub.reshape(-1)

    # precompute these
    XtX = (X_aug.T @ X_aug) / n
    Xty = (X_aug.T @ y_vec) / n
    yty = float(y_vec @ y_vec) / n

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.setParam("TimeLimit", time_limit)
    env.start()

    model = gp.Model(env=env)

    # model variables
    wb = model.addMVar(p + 1, lb=-GRB.INFINITY, name="wb")
    z  = model.addMVar(p, vtype=GRB.BINARY, name="z")

    # obj: (1/n)||y - X_aug wb||^2 = wb'XtX wb - 2 Xty'wb + yty
    model.setObjective(wb @ XtX @ wb - 2 * Xty @ wb + yty, GRB.MINIMIZE)

    # atmost k features
    model.addConstr(z.sum() <= k, name="k_feat_constraint")

    # big-M stuff: wb_j in [-M*z_j, M*z_j]
    model.addConstr(wb[:p] <=  M * z, name="bigM_upper")
    model.addConstr(wb[:p] >= -M * z, name="bigM_lower")

    model.optimize()

    # find solution (p - features)
    w_vals   = wb.X[:p]
    b_val    = float(wb.X[p])
    z_vals   = z.X
    obj_gap  = model.MIPGap if model.SolCount > 0 else float("inf")

    selected = list(np.where(z_vals > 0.5)[0])
    return selected, w_vals, b_val, obj_gap

# just predict with linear model defined by w, b
def predict_mip(X, w, b):
    return X @ w + b

# evaluate MIP predictor by computing MAE, MSE, RMSE, dir accuracy
# caller should first select features and then pass them here
def evaluate_mip(X_test, y_test, w, b):
    preds = predict_mip(X_test, w, b)
    mae   = mean_absolute_error(y_test, preds)
    mse   = mean_squared_error(y_test, preds)
    rmse  = np.sqrt(mse)
    mask  = y_test != 0
    dir_acc = float(np.mean(np.sign(preds[mask]) == np.sign(y_test[mask]))) \
              if mask.sum() > 0 else float("nan")
    return {"mae": mae, "mse": mse, "rmse": rmse, "dir_acc": dir_acc,
            "n_features": int(np.sum(np.abs(w) > 1e-8))}