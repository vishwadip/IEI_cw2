import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path="data/cleaned_ci_cd_logs.csv"):
    df = pd.read_csv(path)

    
    feature_cols = [
        "pipeline_id_enc", "stage_name_enc", "job_name_enc",
        "task_name_enc", "branch_enc", "user_enc", "environment_enc",
        "hour", "weekday", "message_len", "message_has_fail",
        "is_user_unknown", "is_env_unknown", "job_fail_count"
    ]
    X = df[feature_cols]
    y = df["target"]  

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test, feature_cols
