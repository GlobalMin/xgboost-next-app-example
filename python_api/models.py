from pydantic import BaseModel, field_validator, Field
from typing import List, Dict, Any, Optional


class TrainRequest(BaseModel):
    model_config = {"protected_namespaces": ()}

    model_name: str
    csv_filename: str
    target_column: str
    feature_columns: List[str] = Field(..., min_length=1)
    test_size: float = Field(0.2, gt=0, lt=1)
    cv_folds: int = 3
    tune_parameters: bool = True
    early_stopping_rounds: int = 50
    objective: str = "binary:logistic"
    custom_param_grid: Optional[Dict[str, List[Any]]] = None

    @field_validator("custom_param_grid")
    @classmethod
    def validate_custom_param_grid(
        cls, v: Optional[Dict[str, List[Any]]]
    ) -> Optional[Dict[str, List[Any]]]:
        """Validate custom parameter grid if provided"""
        if v is None:
            return v

        from xgb_params import validate_param_grid

        is_valid, errors = validate_param_grid(v)

        if not is_valid:
            raise ValueError(f"Invalid parameter grid: {'; '.join(errors)}")

        return v


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
