from src.data_loader import load_fi2010
from src.labels import process_labels
from src.baselines import random_baseline, momentum_baseline
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report

X_train, y_train = load_fi2010("data/BenchmarkDatasets/Auction/1.Auction_Zscore/Auction_Zscore_Training/Train_Dst_Auction_ZScore_CF_1.txt")
X_test, y_test = load_fi2010("data/BenchmarkDatasets/Auction/1.Auction_Zscore/Auction_Zscore_Testing/Test_Dst_Auction_ZScore_CF_1.txt")

y_train = process_labels(y_train)
y_test = process_labels(y_test)

rand_preds = random_baseline(X_test)
print("\nRandom Baseline")
print("Accuracy:", accuracy_score(y_test, rand_preds))
print("F1 (macro):", f1_score(y_test, rand_preds, average='macro'))
print(classification_report(y_test, rand_preds, target_names=['down', 'stationary', 'up'], zero_division=0))

most_frequent = np.bincount(y_train).argmax()
mom_preds = momentum_baseline(X_test, last_train_label=most_frequent)

print("\nMomentum Baseline")
print("Accuracy:", accuracy_score(y_test, mom_preds))
print("F1 (macro):", f1_score(y_test, mom_preds, average='macro'))
print(classification_report(y_test, mom_preds, target_names=['down', 'stationary', 'up'], zero_division=0))