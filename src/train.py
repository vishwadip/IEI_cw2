
import os
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from src.data_loader import load_data

# --- 1) Remove Azure/auto-populated MLflow/AzureML env vars that cause conflicts ---
for v in (
    "MLFLOW_RUN_ID", "MLFLOW_RUN_NAME", "MLFLOW_TRACKING_URI",
    "MLFLOW_EXPERIMENT_ID", "MLFLOW_EXPERIMENT_NAME",
    "AZUREML_RUN_ID", "AZUREML_RUN_NAME"
):
    os.environ.pop(v, None)

# --- 2) Use a simple sqlite backend in the job working directory (safe & supported) ---
mlflow_db = os.environ.get("MLFLOW_DB_URI", "sqlite:///./mlflow.db")
mlflow.set_tracking_uri(mlflow_db)
print("MLflow tracking URI:", mlflow.get_tracking_uri())

# --- 3) Create or set an experiment name (ensures experiment id matches) ---
EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT_NAME", "ci_cd_failure_prediction")
try:
    mlflow.set_experiment(EXPERIMENT_NAME)
    print("Using MLflow experiment:", EXPERIMENT_NAME)
except Exception as e:
    print("Warning setting experiment:", e)

def main():
    X_train, X_test, y_train, y_test, feature_cols = load_data()

    with mlflow.start_run():
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print("Accuracy:", acc)
        print("F1 Score:", f1)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_text(",".join(feature_cols), "feature_columns.txt")

if __name__ == "__main__":
    main()
