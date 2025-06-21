"""Code generator for standalone XGBoost training scripts."""

import json
from typing import Dict, List
from textwrap import dedent


def generate_training_code(
    csv_filename: str,
    feature_columns: List[str],
    target_column: str,
    test_size: float,
    model_params: Dict,
    preprocessing_artifacts: Dict,
    n_estimators: int,
    objective: str = "binary:logistic",
    eval_metric: str = "auc",
) -> str:
    """Generate standalone Python code to reproduce XGBoost training."""

    # Convert preprocessing artifacts to code-friendly format
    numeric_columns = preprocessing_artifacts.get("numeric_columns", [])
    categorical_columns = preprocessing_artifacts.get("categorical_columns", [])

    code = dedent(f'''
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.impute import SimpleImputer

# Load data
df = pd.read_csv("{csv_filename}")

# Define feature columns
numeric_features = {numeric_columns}
categorical_features = {categorical_columns}
target_column = "{target_column}"

# Extract features and target
feature_columns = numeric_features + categorical_features
X = df[feature_columns]
y = df[target_column]

# Preprocessing pipeline
# Handle numeric features
if numeric_features:
    numeric_imputer = SimpleImputer(strategy="constant", fill_value=-9999)
    X[numeric_features] = numeric_imputer.fit_transform(X[numeric_features])

# Handle categorical features
if categorical_features:
    categorical_imputer = SimpleImputer(strategy="constant", fill_value="missing")
    X[categorical_features] = categorical_imputer.fit_transform(X[categorical_features])
    
    # Encode categorical variables
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X[categorical_features] = encoder.fit_transform(X[categorical_features])

# Ensure all features are float32
X = X.astype(np.float32)

# Encode target if categorical
if y.dtype == "object":
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

# Model parameters
params = {json.dumps(model_params, indent=4)}
params["objective"] = "{objective}"

# Train model
dtrain = xgb.DMatrix(X, label=y)
model = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round={n_estimators}
)

# Save model
model.save_model("xgboost_model.json")
print("Model training completed and saved to xgboost_model.json")
''')

    return code.strip()
