import os
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from src.data_loader import load_data
import joblib

FEATURE_FILE = "feature_columns.txt"

# Try to import Azure ML Run (works when running inside Azure ML)
try:
    from azureml.core import Run
except Exception:
    Run = None

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
    # attempt to get azure run context and print diagnostics
    azure_run = None
    if Run is not None:
        try:
            azure_run = Run.get_context()
            print("DEBUG: Run.get_context() returned:", type(azure_run))
            try:
                print("DEBUG: azure_run.id:", getattr(azure_run, "id", "<no-id-attr>"))
            except Exception as e:
                print("DEBUG: couldn't read azure_run.id:", e)
            print("DEBUG: AZUREML_RUN_ID env:", os.environ.get("AZUREML_RUN_ID"))
        except Exception as e:
            print("DEBUG: Run.get_context() raised:", repr(e))
    else:
        print("DEBUG: azureml.core.Run not available in this environment (local run).")

    # load data
    X_train, X_test, y_train, y_test, feature_cols = load_data()

    with mlflow.start_run():
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        model.fit(X_train, y_train)

        # save model locally for Docker API
        joblib.dump(model, "model.joblib")
        print("Saved model to model.joblib")

        # save feature columns locally for Docker API
        try:
            with open(FEATURE_FILE, "w") as f:
                f.write(",".join(feature_cols))
            print(f"Saved feature column names to {FEATURE_FILE}")
        except Exception as e:
            print(f"WARNING: failed to write {FEATURE_FILE}:", e)

        # predictions / metrics
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print("Accuracy:", acc)
        print("F1 Score:", f1)

        # Log to MLflow (local sqlite)
        try:
            mlflow.log_metric("accuracy", float(acc))
            mlflow.log_metric("f1_score", float(f1))
            print("DEBUG: MLflow metrics logged.")
        except Exception as e:
            print("WARNING: failed to log metrics to MLflow:", e)

        # Also log to Azure ML run so metrics show up in Azure portal
        if azure_run is not None:
            try:
                print("DEBUG: Attempting to log metrics to Azure ML run...")
                azure_run.log("accuracy", float(acc))
                azure_run.log("f1_score", float(f1))
                print("DEBUG: Successfully logged metrics to Azure ML run.")
            except Exception as e:
                print("WARNING: failed to log metrics to Azure ML run:", repr(e))
        else:
            print("DEBUG: azure_run is None — skipping azure_run.log()")

        # Log/save the model and feature list as MLflow artifacts
        try:
            mlflow.sklearn.log_model(model, "model")
            mlflow.log_text(",".join(feature_cols), "feature_columns.txt")
            print("DEBUG: model & feature_columns.txt logged to MLflow.")
        except Exception as e:
            print("WARNING: failed to log model/artifacts to MLflow:", e)


if __name__ == "__main__":
    main()
