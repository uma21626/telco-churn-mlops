import mlflow
import pandas as pd
import mlflow.xgboost
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
from sklearn.metrics import roc_auc_score

def train_model(df: pd.DataFrame, target_col: str):
    """
    Trains an XGBoost model and logs with MLflow
    Args:
        df (pd.DataFrame): Feature dataset
        target_col (str): Name of the target column
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )

    with mlflow.start_run():
        # Train model
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        preds_proba = model.predict_proba(X_test)[:, 1]

        #  Calculate metrics correctly
        acc       = accuracy_score(y_test, preds)
        rec       = recall_score(y_test, preds)
        prec      = precision_score(y_test, preds)
        f1        = f1_score(y_test, preds)
        roc_auc   = roc_auc_score(y_test, preds_proba)

        #  Log metrics to MLflow
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("roc_auc", roc_auc)

        #  Log parameters
        mlflow.log_param("n_estimators", 300)
        mlflow.log_param("learning_rate", 0.1)
        mlflow.log_param("max_depth", 6)

        #  Log model
        mlflow.xgboost.log_model(model, "model")

        #  Log dataset
        train_ds = mlflow.data.from_pandas(df, source="training_data")
        mlflow.log_input(train_ds, context="training")

        print(f"Model trained successfully!")
        print(f"  Accuracy  : {acc:.4f}")
        print(f"  Recall    : {rec:.4f}")
        print(f"  Precision : {prec:.4f}")
        print(f"  F1 Score  : {f1:.4f}")
        print(f"  ROC-AUC   : {roc_auc:.4f}")

    return model
