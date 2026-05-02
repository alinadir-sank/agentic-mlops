# train_dummy.py
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("production-model-evals")

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y)

model = RandomForestClassifier()
model.fit(X_train, y_train)
acc = accuracy_score(y_test, model.predict(X_test))

with mlflow.start_run(run_name="fraud-classifier-v2"):
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("drift_score", 0.15)      # fake but realistic
    mlflow.log_metric("latency_ms", 120.0)
    mlflow.log_metric("error_rate", 0.02)
    mlflow.sklearn.log_model(model, "model")

print(f"Logged run with accuracy={acc:.3f}")