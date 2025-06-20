"""Utility functions for XGBoost modeling workflow"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from typing import Dict, Any, Tuple, List, Optional
import os
from itertools import product


from config import UPLOAD_DIR, MODEL_DIR
from logging_config import get_logger

# Get logger for this module
logger = get_logger(__name__)


def load_dataset(filename: str) -> pd.DataFrame:
    """Load dataset from uploaded CSV file"""
    file_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {filename}")

    return pd.read_csv(file_path)


def preprocess_data(
    df: pd.DataFrame, feature_columns: List[str], target_column: str
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    """Preprocess data for XGBoost training with improved handling of categorical and missing values

    Returns:
        Tuple of (X_processed, y_processed, preprocessing_artifacts)
    """
    X = df[feature_columns].copy()
    y = df[target_column].copy()

    # Handle missing values in target
    if y.isnull().any():
        raise ValueError("Target column contains missing values")

    # Separate numeric and categorical columns
    numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = X.select_dtypes(include=["object"]).columns.tolist()

    # Store preprocessing artifacts for potential inverse transformation
    preprocessing_artifacts = {
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "encoders": {},
        "imputers": {},
    }

    # Handle missing values for numeric features using median imputation
    if numeric_columns:
        numeric_imputer = SimpleImputer(strategy="median")
        X[numeric_columns] = numeric_imputer.fit_transform(X[numeric_columns])
        preprocessing_artifacts["imputers"]["numeric"] = numeric_imputer

    # Handle missing values for categorical features using constant imputation
    if categorical_columns:
        # For categorical features, fill missing with a constant
        categorical_imputer = SimpleImputer(strategy="constant", fill_value="missing")
        X[categorical_columns] = categorical_imputer.fit_transform(
            X[categorical_columns]
        )
        preprocessing_artifacts["imputers"]["categorical"] = categorical_imputer

        # Use OrdinalEncoder for categorical variables
        # OrdinalEncoder handles unknown categories better than LabelEncoder
        ordinal_encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,  # Unknown categories will be encoded as -1
        )
        X[categorical_columns] = ordinal_encoder.fit_transform(X[categorical_columns])
        preprocessing_artifacts["encoders"]["categorical"] = ordinal_encoder

        # Store individual column names with their categories for interpretability
        for i, col in enumerate(categorical_columns):
            preprocessing_artifacts["encoders"][col] = {
                "categories": ordinal_encoder.categories_[i].tolist()  # type: ignore
            }

    # Encode target if categorical
    if y.dtype == "object":
        le_y = LabelEncoder()
        y_encoded = le_y.fit_transform(y)
        y = pd.Series(y_encoded, index=y.index, name=y.name)  # type: ignore
        preprocessing_artifacts["encoders"]["_target"] = le_y

    # Ensure all columns are numeric after preprocessing
    X = X.astype(np.float32)

    return X, y, preprocessing_artifacts


def create_train_test_split(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create stratified train-test split"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


def generate_parameter_grid(base_grid: Dict[str, List]) -> List[Dict[str, Any]]:
    """Generate parameter combinations from grid using itertools"""
    param_names = list(base_grid.keys())
    param_values = [base_grid[name] for name in param_names]

    return [dict(zip(param_names, values)) for values in product(*param_values)]


def train_single_model(
    dtrain: xgb.DMatrix,
    params: Dict[str, Any],
    num_rounds: int,
    evals: Optional[List] = None,
    early_stopping_rounds: Optional[int] = None,
) -> xgb.Booster:
    """Train a single XGBoost model"""
    return xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_rounds,
        evals=evals,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=False,
    )


def cross_validate_params(
    dtrain: xgb.DMatrix,
    params: Dict[str, Any],
    cv_folds: int,
    num_rounds: int,
    early_stopping_rounds: int,
) -> Tuple[float, float, int]:
    """Run cross-validation for a parameter set

    Returns:
        Tuple of (mean_score, std_score, optimal_rounds)
    """
    cv_results = xgb.cv(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_rounds,
        nfold=cv_folds,
        stratified=True,
        seed=42,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=False,
    )

    if isinstance(cv_results, pd.DataFrame) and len(cv_results) > 0:
        mean_score = cv_results["test-auc-mean"].iloc[-1]
        std_score = cv_results["test-auc-std"].iloc[-1]
        optimal_rounds = len(cv_results)
    else:
        mean_score = 0.5
        std_score = 0.0
        optimal_rounds = num_rounds

    return mean_score, std_score, optimal_rounds


def calculate_feature_importance(
    model: xgb.Booster, feature_names: List[str]
) -> Dict[str, float]:
    """Extract feature importance from trained model"""
    # Try different importance types to get meaningful values
    importance_dict = model.get_score(
        importance_type="gain"
    )  # Use 'gain' instead of 'weight'

    # If gain doesn't work, try weight
    if not importance_dict:
        importance_dict = model.get_score(importance_type="weight")

    # If still empty, try cover
    if not importance_dict:
        importance_dict = model.get_score(importance_type="cover")

    feature_importance = {}
    for feature in feature_names:
        # Try to get importance by feature name directly first
        importance_value = importance_dict.get(feature, 0.0)

        # If not found, try the f{i} format as fallback
        if importance_value == 0.0:
            feature_index = (
                feature_names.index(feature) if feature in feature_names else -1
            )
            if feature_index >= 0:
                importance_value = importance_dict.get(f"f{feature_index}", 0.0)

        if isinstance(importance_value, list):
            importance_value = importance_value[0] if importance_value else 0.0
        feature_importance[feature] = float(importance_value)

    return feature_importance


def calculate_lift_chart(
    y_true: np.ndarray, y_pred_proba: np.ndarray, n_bins: int = 10
) -> List[Dict[str, Any]]:
    """Calculate lift chart data for model evaluation"""
    df = pd.DataFrame({"y_true": y_true, "y_pred_proba": y_pred_proba})

    # Sort by predicted probability and create bins
    df = df.sort_values("y_pred_proba")
    df["bin"] = pd.qcut(df["y_pred_proba"], n_bins, labels=False, duplicates="drop")

    # Calculate metrics for each bin
    lift_data = []
    for bin_num in range(n_bins):
        bin_data = df[df["bin"] == bin_num]
        if len(bin_data) > 0:
            lift_data.append(
                {
                    "bin": int(bin_num + 1),
                    "avg_prediction": float(bin_data["y_pred_proba"].mean()),
                    "actual_rate": float(bin_data["y_true"].mean()),
                    "count": int(len(bin_data)),
                }
            )

    return lift_data


def save_model(model: xgb.Booster, project_id: str) -> str:
    """Save XGBoost model to file"""
    model_path = os.path.join(MODEL_DIR, f"{project_id}.json")
    model.save_model(model_path)
    return model_path


def prepare_results(
    test_auc: float,
    cv_auc: float,
    feature_importance: Dict[str, float],
    lift_data: List[Dict],
    best_params: Dict[str, Any],
    n_estimators: int,
) -> Dict[str, Any]:
    """Prepare final results dictionary"""
    return {
        "test_auc": float(test_auc),
        "cv_auc": float(cv_auc),
        "feature_importance": feature_importance,
        "lift_chart_data": lift_data,
        "best_params": best_params,
        "n_estimators_used": int(n_estimators),
    }


def tune_hyperparameters(
    dtrain: xgb.DMatrix,
    base_params: Dict[str, Any],
    param_grid: Dict[str, List],
    cv_folds: int,
    early_stopping_rounds: int,
    project_id: str,
) -> Tuple[Dict[str, Any], float, int]:
    """Tune hyperparameters using grid search

    Returns:
        Tuple of (best_params, best_score, best_n_estimators)
    """
    param_combinations = generate_parameter_grid(param_grid)
    logger.info(
        f"Testing {len(param_combinations)} parameter combinations",
        extra={"project_id": project_id},
    )

    best_score = -np.inf
    best_params = {}
    best_n_estimators = 100

    for idx, params in enumerate(param_combinations):
        cv_params = {**base_params, **params}

        mean_score, _, optimal_rounds = cross_validate_params(
            dtrain, cv_params, cv_folds, 500, early_stopping_rounds
        )

        if mean_score > best_score:
            best_score = mean_score
            best_params = params.copy()
            best_n_estimators = optimal_rounds

        # Log progress every 5 combinations
        if (idx + 1) % 5 == 0:
            logger.info(
                f"Progress: {idx + 1}/{len(param_combinations)}, Best AUC: {best_score:.4f}",
                extra={"project_id": project_id},
            )

    logger.info(
        f"Best params: {best_params}, CV AUC: {best_score:.4f}, Trees: {best_n_estimators}",
        extra={"project_id": project_id},
    )

    return best_params, best_score, best_n_estimators
