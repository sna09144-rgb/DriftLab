from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import numpy as np

def build_model(name="logistic_regression", seed=42):
    if name != "logistic_regression":
        raise ValueError(f"Неизвестная модель: {name}")
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=seed))
    ])

def fit_model(X, y, name="logistic_regression", seed=42):
    model = build_model(name, seed)
    model.fit(np.asarray(X), np.asarray(y))
    return model

def predict_model(model, X):
    return model.predict(np.asarray(X)).astype(int)

def positive_scores(model, X):
    return model.predict_proba(np.asarray(X))[:, 1]
