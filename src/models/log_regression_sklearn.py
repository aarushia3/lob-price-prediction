from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=1000, solver='lbfgs')
    model.fit(X_train, y_train)
    return model

def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    return {
        "acc": accuracy_score(y_test, preds),
        "f1": f1_score(y_test, preds, average='macro')
    }