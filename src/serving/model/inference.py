"""
INFERENCE PIPELINE - Production ML Model Serving with Feature Consistency
=========================================================================

This module provides the core inference functionality for the Telco Churn prediction model.
It ensures that serving-time feature transformations exactly match training-time transformations,
which is CRITICAL for model accuracy in production.

Key Responsibilities:
1. Load MLflow-logged model and feature metadata from training
2. Apply identical feature transformations as used during training
3. Ensure correct feature ordering for model input
4. Convert model predictions to user-friendly output

CRITICAL PATTERN: Training/Serving Consistency
- Uses fixed BINARY_MAP for deterministic binary encoding
- Applies same one-hot encoding with drop_first=True
- Maintains exact feature column order from training
- Handles missing/new categorical values gracefully

Production Deployment:
- MODEL_DIR points to containerized model artifacts
- Feature schema loaded from training-time artifacts
- Optimized for single-row inference (real-time serving)
"""

import os
import glob
import pandas as pd
import mlflow

# === MODEL LOADING CONFIGURATION ===
# Works in Docker (/app/model) and Windows (auto-find mlruns)

# Always define MODEL_DIR first with a default
MODEL_DIR = "/app/model"

if not os.path.exists(MODEL_DIR):
    # Auto-find on Windows locally
    MLRUNS_DIR = os.path.join(os.getcwd(), "mlruns")
    MODEL_DIR = None
    for root, dirs, files in os.walk(MLRUNS_DIR):
        if "MLmodel" in files:
            MODEL_DIR = root
            break

if MODEL_DIR is None:
    raise Exception("Could not find model directory!")

try:
    model = mlflow.pyfunc.load_model(MODEL_DIR)
    print(f"Model loaded successfully from {MODEL_DIR}")
except Exception as e:
    print(f"Failed to load model from {MODEL_DIR}: {e}")
    try:
        local_model_paths = glob.glob("./mlruns/*/*/artifacts/model")
        if local_model_paths:
            latest_model = max(local_model_paths, key=os.path.getmtime)
            model = mlflow.pyfunc.load_model(latest_model)
            MODEL_DIR = latest_model
            print(f"Fallback: Loaded model from {latest_model}")
        else:
            raise Exception("No model found in local mlruns")
    except Exception as fallback_error:
        raise Exception(f"Failed to load model: {e}. Fallback failed: {fallback_error}")

# === FEATURE SCHEMA LOADING ===
# Works in Docker (/app/model/feature_columns.txt) and Windows (auto-find)

try:
    if os.path.exists("/app/model/feature_columns.txt"):
        feature_file = "/app/model/feature_columns.txt"  # Docker path
    elif os.path.exists(os.path.join(os.getcwd(), "src", "serving", "model", "artifacts", "feature_columns.txt")):
        feature_file = os.path.join(os.getcwd(), "src", "serving", "model", "artifacts", "feature_columns.txt")    
    else:
        # Auto-find on Windows locally
        MLRUNS_DIR = os.path.join(os.getcwd(), "mlruns")
        feature_file = None
        for root, dirs, files in os.walk(MLRUNS_DIR):
            if "feature_columns.txt" in files:
                feature_file = os.path.join(root, "feature_columns.txt")
                break
        if feature_file is None:
            raise Exception("feature_columns.txt not found!")
    with open(feature_file) as f:
        FEATURE_COLS = [ln.strip() for ln in f if ln.strip()]
    print(f"Loaded {len(FEATURE_COLS)} feature columns from training")
except Exception as e:
    raise Exception(f"Failed to load feature columns: {e}")

# === FEATURE TRANSFORMATION CONSTANTS ===
# CRITICAL: These mappings must exactly match those used in training

BINARY_MAP = {
    "gender": {"Female": 0, "Male": 1},
    "Partner": {"No": 0, "Yes": 1},
    "Dependents": {"No": 0, "Yes": 1},
    "PhoneService": {"No": 0, "Yes": 1},
    "PaperlessBilling": {"No": 0, "Yes": 1},
}

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

def _serve_transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply identical feature transformations as used during model training.
    """
    df = df.copy()

    # Clean column names
    df.columns = df.columns.str.strip()

    # STEP 1: Numeric Type Coercion
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df[c] = df[c].fillna(0)

    # STEP 2: Binary Feature Encoding
    for c, mapping in BINARY_MAP.items():
        if c in df.columns:
            df[c] = (
                df[c]
                .astype(str)
                .str.strip()
                .map(mapping)
                .astype("Int64")
                .fillna(0)
                .astype(int)
            )

    # STEP 3: One-Hot Encoding
    obj_cols = [c for c in df.select_dtypes(include=["object"]).columns]
    if obj_cols:
        df = pd.get_dummies(df, columns=obj_cols, drop_first=True)

    # STEP 4: Boolean to Integer
    bool_cols = df.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)

    # STEP 5: Feature Alignment
    df = df.reindex(columns=FEATURE_COLS, fill_value=0)

    return df

def predict(input_dict: dict) -> str:
    """
    Main prediction function for customer churn inference.
    """
    df = pd.DataFrame([input_dict])
    df_enc = _serve_transform(df)

    try:
        preds = model.predict(df_enc)

        if hasattr(preds, "tolist"):
            preds = preds.tolist()

        if isinstance(preds, (list, tuple)) and len(preds) == 1:
            result = preds[0]
        else:
            result = preds

    except Exception as e:
        raise Exception(f"Model prediction failed: {e}")

    if result == 1:
        return "Likely to churn"
    else:
        return "Not likely to churn"
