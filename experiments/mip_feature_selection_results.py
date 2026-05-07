# run from project root:
# python experiments/mip_feature_selection_results.py

# mip feature selection experiment over all 9 fi-2010 folds
# for each fold, run mip feat select with k = 5, 10, 20 and compare selected feat sets against top-k by absolute correlation with y_train
# then fit ridge on selected features and evaluate on test set

import numpy as np
from src.data_loader import load_fi2010_regression
from src.models.ridge_regression import train_ridge, evaluate
from src.models.mip_feature_selection import select_features_mip, evaluate_mip

K_VALUES   = [5, 10, 20]
TIME_LIMIT = 60          # seconds per MIP solve
MAX_SAMPLES = 5000       # rows passed to MIP

results = {k: [] for k in K_VALUES}

for fold in range(1, 10):
    train_path = f"data/BenchmarkDatasets/NoAuction/1.NoAuction_Zscore/NoAuction_Zscore_Training/Train_Dst_NoAuction_ZScore_CF_{fold}.txt"
    test_path = f"data/BenchmarkDatasets/NoAuction/1.NoAuction_Zscore/NoAuction_Zscore_Testing/Test_Dst_NoAuction_ZScore_CF_{fold}.txt"
    
    X_train, y_train = load_fi2010_regression(train_path, horizon=10)
    X_test,  y_test  = load_fi2010_regression(test_path,  horizon=10)

    # compute correlation of each feature with y_train and rank by absolute correlation
    with np.errstate(invalid='ignore'):
        correlations = np.array([
            np.corrcoef(X_train[:, j], y_train)[0, 1]
            for j in range(X_train.shape[1])
        ])
    
    # handle nan corrs (0 variance - just say corr = 0)
    correlations = np.nan_to_num(correlations, nan=0.0)
    corr_ranking = np.argsort(np.abs(correlations))[::-1]

    for k in K_VALUES:
        print(f"Fold {fold}, k={k} ... ", end="", flush=True)

        # mip selection + evaluaton
        selected_mip, w_mip, b_mip, gap = select_features_mip(
            X_train, y_train, k=k, time_limit=TIME_LIMIT, max_samples=MAX_SAMPLES
        )
        mip_res = evaluate_mip(X_test, y_test, w_mip, b_mip)

        # top-k correlation training + evaluation
        corr_feats = list(corr_ranking[:k])
        ridge_corr = train_ridge(X_train[:, corr_feats], y_train, alpha=1.0)
        corr_res   = evaluate(ridge_corr, X_test[:, corr_feats], y_test)

        print(
            f"MIP(gap={gap:.3%}, DirAcc={mip_res['dir_acc']:.3f}) | "
            f"Corr(DirAcc={corr_res['dir_acc']:.3f}) "
        )

        results[k].append({
            "fold":        fold,
            "mip":         mip_res,
            "corr":        corr_res,
            "mip_gap":     gap,
            "mip_feats":   selected_mip,
            "corr_feats":  corr_feats,
        })

# get claude to fix this summary printout
print("\n── Summary (averaged over 9 folds) ──────────────────────────────────")
for k in K_VALUES:
    r = results[k]
    print(f"\nk = {k}")
    for method in ["mip", "corr"]:
        avg_rmse    = np.mean([x[method]["rmse"]    for x in r])
        avg_dir_acc = np.mean([x[method]["dir_acc"] for x in r])
        print(f"  {method:5s}: RMSE={avg_rmse:.6f}  DirAcc={avg_dir_acc:.3f}")
    avg_gap = np.mean([x["mip_gap"] for x in r])
    print(f"  MIP avg optimality gap: {avg_gap:.3%}")