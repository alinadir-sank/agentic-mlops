# # train_dummy.py
# import mlflow
# import mlflow.sklearn
# from sklearn.datasets import load_iris
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# import numpy as np

# mlflow.set_tracking_uri("http://localhost:5000")
# mlflow.set_experiment("production-model-evals")

# X, y = load_iris(return_X_y=True)
# X_train, X_test, y_train, y_test = train_test_split(X, y)

# model = RandomForestClassifier()
# model.fit(X_train, y_train)
# acc = accuracy_score(y_test, model.predict(X_test))

# with mlflow.start_run(run_name="fraud-classifier-v2"):
#     mlflow.log_metric("accuracy", acc)
#     mlflow.log_metric("drift_score", 0.15)      # fake but realistic
#     mlflow.log_metric("latency_ms", 120.0)
#     mlflow.log_metric("error_rate", 0.02)
#     mlflow.sklearn.log_model(model, "model")

# print(f"Logged run with accuracy={acc:.3f}")



import mlflow
import mlflow.sklearn
import numpy as np
import time
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.stats import wasserstein_distance

# --- ALIGN WITH YOUR .env CONFIG ---
# Ensure these match your DEFAULT_MODEL_ID and MLFLOW_EXPERIMENT_NAME
MODEL_ID = "your-model-name" 
ENVIRONMENT = "production"
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("production-model-evals") 

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=10, random_state=42, stratify=y
)
X_REFERENCE = X_train.copy()
model = LogisticRegression(max_iter=1000, random_state=42)

def compute_drift(X_ref, X_curr):
    return float(np.mean([wasserstein_distance(X_ref[:, i], X_curr[:, i]) for i in range(X_ref.shape[1])]))

cycle = 0
while True:
    cycle += 1
    current_X_test = X_test.copy()
    current_y_test = y_test.copy()

    # --- CHAOS INJECTION FOR AGENT TESTING ---
    if 5 < cycle <= 10:
        # DATA DRIFT: Trigger 'minor' or 'major' severity
        current_X_test[:, 2] = current_X_test[:, 2] * 2.5 
    elif cycle > 10:
        # CONCEPT DRIFT: Trigger 'critical' severity
        current_y_test = np.where(current_y_test == 0, 1, current_y_test)

    model.fit(X_train, y_train)
    preds = model.predict(current_X_test)
    
    # --- METRICS MATCHING YOUR AGENT REQUIREMENTS ---
    metrics = {
        "accuracy": round(accuracy_score(current_y_test, preds), 3),
        "drift_score": round(compute_drift(X_REFERENCE, current_X_test), 3),
        "latency_ms": round(np.random.uniform(200, 1200), 2), # Random latency jitter
        "error_rate": round(float((model.predict_proba(current_X_test).max(axis=1) < 0.6).mean()), 3)
    }

    with mlflow.start_run(run_name=f"{MODEL_ID}-cycle-{cycle}"):
        mlflow.log_metrics(metrics)
        mlflow.set_tag("model_id", MODEL_ID)
        mlflow.set_tag("environment", ENVIRONMENT)
        mlflow.sklearn.log_model(model, "model")

    print(f"Cycle {cycle} logged. Accuracy: {metrics['accuracy']} | Drift: {metrics['drift_score']}")
    time.sleep(30) # Interval for your monitor_agent to scrape