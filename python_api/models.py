from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "xgboost_models.db")

class TrainRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    
    model_name: str
    csv_filename: str
    target_column: str
    feature_columns: List[str]
    test_size: float = 0.2
    cv_folds: int = 3
    tune_parameters: bool = True
    early_stopping_rounds: int = 50
    objective: str = "binary:logistic"

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

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn