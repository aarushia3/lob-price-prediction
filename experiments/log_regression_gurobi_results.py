from src.data_loader import load_fi2010
from src.labels import process_labels
from src.models.log_regression_gurobi import train_logistic_gurobi, evaluate
import numpy as np

results = []
for fold in range(1, 10):
    train_path = f"data/BenchmarkDatasets/NoAuction/1.NoAuction_Zscore/NoAuction_Zscore_Training/Train_Dst_NoAuction_ZScore_CF_{fold}.txt"
    test_path = f"data/BenchmarkDatasets/NoAuction/1.NoAuction_Zscore/NoAuction_Zscore_Testing/Test_Dst_NoAuction_ZScore_CF_{fold}.txt"

    X_train, y_train = load_fi2010(train_path)
    X_test, y_test = load_fi2010(test_path)

    y_train = process_labels(y_train)
    y_test = process_labels(y_test)

    W, losses = train_logistic_gurobi(X_train, y_train, lr=0.01, num_iters=500)
    eval_results = evaluate(X_test, y_test, W)

    results.append({"fold": fold, **eval_results})
    print(f"Fold {fold}: Acc={eval_results['acc']:.4f}, F1={eval_results['f1']:.4f}")

avg_acc = np.mean([r['acc'] for r in results])
avg_f1  = np.mean([r['f1'] for r in results])
print(f"\nAverage: Acc={avg_acc:.4f}, F1={avg_f1:.4f}")