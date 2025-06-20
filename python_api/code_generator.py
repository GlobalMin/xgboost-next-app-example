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
    eval_metric: str = "auc"
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

# Load data
df = pd.read_csv("{csv_filename}")
X = df[{numeric_columns} + {categorical_columns}]
y = df["{target_column}"]

# Preprocessing
X[{numeric_columns}] = X[{numeric_columns}].fillna(-9999)
if {categorical_columns}:
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X[{categorical_columns}] = encoder.fit_transform(X[{categorical_columns}]).fillna(-9999)
X = X.astype(np.float32)

if y.dtype == "object":
    y = LabelEncoder().fit_transform(y)

# Train
params = {json.dumps(model_params, indent=4)}
params.update({{"objective": "{objective}", "eval_metric": "{eval_metric}", "seed": 42, "nthread": -1}})

model = xgb.train(
    params,
    xgb.DMatrix(X, label=y),
    num_boost_round={n_estimators}
)
model.save_model("xgboost_model.json")
''')

    return code.strip()