# run from project root:
# python experiments/regression_results.py

# compute regression results for zero-return baseline, lagged-return baseline and ridge regression over all features for all 9 fi-2010 folds

from src.data_loader import load_fi2010_regression
from src.baselines import zero_return_baseline, lagged_return_baseline
from src.models.ridge_regression import train_ridge, evaluate
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

# helper to eval regression predictions and compute MAE, MSE, RMSE, dir accuracy
def eval_preds(preds, y_test):
    mae  = mean_absolute_error(y_test, preds)
    mse  = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    dir_acc = np.mean(np.sign(preds) == np.sign(y_test))
    return {"mae": mae, "rmse": rmse, "dir_acc": dir_acc}

results = []

for fold in range(1, 10):
    train_path = f"data/BenchmarkDatasets/NoAuction/1.NoAuction_Zscore/NoAuction_Zscore_Training/Train_Dst_NoAuction_ZScore_CF_{fold}.txt"
    test_path = f"data/BenchmarkDatasets/NoAuction/1.NoAuction_Zscore/NoAuction_Zscore_Testing/Test_Dst_NoAuction_ZScore_CF_{fold}.txt"

    X_train, y_train = load_fi2010_regression(train_path, horizon=10)
    X_test,  y_test  = load_fi2010_regression(test_path,  horizon=10)

    # zero-return baseline
    zero_preds    = zero_return_baseline(X_test)
    
    # lagged-return baseline, scalar needs to be converted to preds array
    lagged_scalar = lagged_return_baseline(y_train)
    lagged_preds  = np.full(len(X_test), lagged_scalar)

    # evaluate baselines
    zero_res   = eval_preds(zero_preds,   y_test)
    lagged_res = eval_preds(lagged_preds, y_test)

    # ridge regression on all features + evaluation
    ridge_model  = train_ridge(X_train, y_train, alpha=1.0)
    ridge_res    = evaluate(ridge_model, X_test, y_test)

    print(
        f"Fold {fold}: "
        f"Zero(DirAcc={zero_res['dir_acc']:.3f}) | "
        f"Lagged(DirAcc={lagged_res['dir_acc']:.3f}) | "
        f"Ridge(RMSE={ridge_res['rmse']:.6f}, DirAcc={ridge_res['dir_acc']:.3f}) "
    )

    results.append({
        "fold":       fold,
        "zero":       zero_res,
        "lagged":     lagged_res,
        "ridge":      ridge_res,
    })

# helper to compute average of a subkey (e.g. dir_acc) across all folds
def avg(key, subkey):
    return np.mean([r[key][subkey] for r in results])

print("\n── Average Results ──────────────────────────────────────────────────")
print(f"Zero   :  DirAcc={avg('zero','dir_acc'):.3f}")
print(f"Lagged :  DirAcc={avg('lagged','dir_acc'):.3f}")
print(f"Ridge  :  RMSE={avg('ridge','rmse'):.6f}  DirAcc={avg('ridge','dir_acc'):.3f}")