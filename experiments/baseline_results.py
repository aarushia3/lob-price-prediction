from src.data_loader import load_fi2010
from src.labels import process_labels
from src.baselines import random_baseline, momentum_baseline
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

results = []
for fold in range(1, 10):
    train_path = f"data/BenchmarkDatasets/NoAuction/1.NoAuction_Zscore/NoAuction_Zscore_Training/Train_Dst_NoAuction_ZScore_CF_{fold}.txt"
    test_path = f"data/BenchmarkDatasets/NoAuction/1.NoAuction_Zscore/NoAuction_Zscore_Testing/Test_Dst_NoAuction_ZScore_CF_{fold}.txt"
    
    X_train, y_train = load_fi2010(train_path)
    X_test, y_test = load_fi2010(test_path)
    
    y_train = process_labels(y_train)
    y_test = process_labels(y_test)
    
    # print("Class distribution (test):", np.bincount(y_test))
    # print("Class ratio:", np.bincount(y_test) / len(y_test))
    
    rand_preds = random_baseline(X_test)
    most_frequent = np.bincount(y_train).argmax()
    mom_preds = momentum_baseline(X_test, last_train_label=most_frequent)
    
    results.append({
        "fold": fold,
        "random_acc": accuracy_score(y_test, rand_preds),
        "random_f1": f1_score(y_test, rand_preds, average='macro'),
        "momentum_acc": accuracy_score(y_test, mom_preds),
        "momentum_f1": f1_score(y_test, mom_preds, average='macro')
    })

print("Baseline Results:")
for res in results:
    print(f"Fold {res['fold']}: Random Acc={res['random_acc']:.4f}, Random F1={res['random_f1']:.4f}, "
          f"Momentum Acc={res['momentum_acc']:.4f}, Momentum F1={res['momentum_f1']:.4f}")
    
avg_rand_acc = np.mean([r['random_acc'] for r in results])
avg_rand_f1  = np.mean([r['random_f1'] for r in results])
avg_mom_acc  = np.mean([r['momentum_acc'] for r in results])
avg_mom_f1   = np.mean([r['momentum_f1'] for r in results])

print(f"\nAverage: Random Acc={avg_rand_acc:.4f}, Random F1={avg_rand_f1:.4f}, "
      f"Momentum Acc={avg_mom_acc:.4f}, Momentum F1={avg_mom_f1:.4f}")