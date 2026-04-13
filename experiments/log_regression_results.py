from src.data_loader import load_fi2010
from src.labels import process_labels
from src.models.log_regression_sklearn import train_logistic_regression, evaluate
import numpy as np

results = []
for fold in range(1, 10):
    train_path = f"data/BenchmarkDatasets/NoAuction/1.NoAuction_Zscore/NoAuction_Zscore_Training/Train_Dst_NoAuction_ZScore_CF_{fold}.txt"
    test_path = f"data/BenchmarkDatasets/NoAuction/1.NoAuction_Zscore/NoAuction_Zscore_Testing/Test_Dst_NoAuction_ZScore_CF_{fold}.txt"
    
    X_train, y_train = load_fi2010(train_path)
    X_test, y_test = load_fi2010(test_path)
    
    y_train = process_labels(y_train)
    y_test = process_labels(y_test)
    
    model = train_logistic_regression(X_train, y_train)
    eval_results = evaluate(model, X_test, y_test)
    
    results.append({
        "fold": fold,
        "log_acc": eval_results["acc"],
        "log_f1": eval_results["f1"]
    })

print("Logistic Regression Results:")
for res in results:
    print(f"Fold {res['fold']}: Log Acc={res['log_acc']:.4f}, Log F1={res['log_f1']:.4f}")

avg_log_acc = np.mean([r['log_acc'] for r in results])
avg_log_f1  = np.mean([r['log_f1'] for r in results])
print(f"\nAverage: Log Acc={avg_log_acc:.4f}, Log F1={avg_log_f1:.4f}")