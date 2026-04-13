import numpy as np
import gurobipy as gp
from gurobipy import GRB

def softmax(Z):
    Z = Z - Z.max(axis=1, keepdims=True)  # numerical stability
    exp_Z = np.exp(Z)
    return exp_Z / exp_Z.sum(axis=1, keepdims=True)

def compute_gradient(X, y, W):
    n, K = X.shape[0], W.shape[1]
    P = softmax(X @ W)          # (n, K) predicted probabilities
    Y_one_hot = np.zeros((n, K))
    Y_one_hot[np.arange(n), y] = 1
    grad = X.T @ (P - Y_one_hot) / n   # (p, K)
    return grad

def compute_loss(X, y, W):
    n, K = X.shape[0], W.shape[1]
    P = softmax(X @ W)
    log_probs = np.log(P[np.arange(n), y] + 1e-9)
    return -np.mean(log_probs)

def train_logistic_gurobi(X_train, y_train, lr=0.01, num_iters=500, tol=1e-6):
    _, p = X_train.shape
    K = len(np.unique(y_train))  # number of classes

    W = np.zeros((p, K))  # weight matrix (p features x K classes)
    losses = []

    for i in range(num_iters):
        loss = compute_loss(X_train, y_train, W)
        losses.append(loss)

        grad = compute_gradient(X_train, y_train, W)

        # solve for W_new that minimizes ||W - (W - lr*grad)||^2
        W_target = W - lr * grad  # (p, K) target after gradient step

        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.start()
        model = gp.Model(env=env)

        # Flatten W into decision variables
        w_vars = model.addMVar(shape=(p * K,), lb=-GRB.INFINITY, name="w")

        # Minimize ||w - w_target||^2
        w_target_flat = W_target.flatten()
        diff = w_vars - w_target_flat
        model.setObjective(diff @ diff, GRB.MINIMIZE)
        model.optimize()

        W = w_vars.X.reshape(p, K)

        # Convergence check
        if i > 0 and abs(losses[-1] - losses[-2]) < tol:
            print(f"Converged at iteration {i}")
            break

    return W, losses

def predict(X, W):
    probs = softmax(X @ W)
    return np.argmax(probs, axis=1)

def evaluate(X_test, y_test, W):
    from sklearn.metrics import accuracy_score, f1_score
    preds = predict(X_test, W)
    return {
        "acc": accuracy_score(y_test, preds),
        "f1": f1_score(y_test, preds, average='macro')
    }