from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import pandas as pd
import xgboost as xgb


class TrainRequest(BaseModel):
    model_config = {"protected_namespaces": ()}

    model_name: str
    csv_filename: str
    target_column: str
    feature_columns: List[str] = Field(..., min_length=1)
    test_size: float = Field(0.2, gt=0, lt=1)
    cv_folds: int = 3
    early_stopping_rounds: int = 50
    objective: str = "binary:logistic"
    custom_param_grid: Optional[Dict[str, List[Any]]] = None


class ModelInfo(BaseModel):
    model_config = {"protected_namespaces": ()}

    id: int
    name: str
    created_at: str
    csv_filename: str
    target_column: str
    feature_columns: List[str]
    model_params: Dict[str, Any]
    auc_score: Optional[float] = None
    accuracy: Optional[float] = None
    feature_importance: Optional[Dict[str, float]] = None
    confusion_matrix: Optional[List[List[int]]] = None
    status: str


class DataPreparationResult(BaseModel):
    """Container for prepared training data"""

    model_config = {"arbitrary_types_allowed": True}

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    dtrain: xgb.DMatrix
    dtest: xgb.DMatrix
    preprocessing_artifacts: Dict[str, Any]
    dataset_info: Dict[str, Any]


class TrainingResult(BaseModel):
    """Container for training results"""

    model_config = {"arbitrary_types_allowed": True}

    model: xgb.Booster
    best_params: Dict[str, Any]
    cv_auc: float
    best_n_estimators: int
