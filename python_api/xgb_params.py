"""XGBoost parameter definitions and validation - simplified version"""

from typing import Dict, List, Any

# Parameters that are allowed for tuning via the GUI
# Each parameter should be submitted as a list of values
ALLOWED_TUNING_PARAMS = {
    "learning_rate",
    "eta",  # alias for learning_rate
    "alpha",
    "reg_alpha",  # alias for alpha
    "lambda",
    "reg_lambda",  # alias for lambda
    "max_depth",
    "min_child_weight",
    "subsample",
    "colsample_bytree",
}

# Default parameter grid for hyperparameter tuning
DEFAULT_PARAM_GRID = {
    "max_depth": [3, 4, 5],
    "learning_rate": [0.01, 0.1],
    "lambda": [0, 0.5, 1.0],
    "subsample": [0.8],
    "colsample_bytree": [0.8],
}


def validate_param_grid(param_grid: Dict[str, List[Any]]) -> tuple[bool, List[str]]:
    """
    Validate that parameter grid only contains allowed parameters.
    Returns (is_valid, list_of_errors)
    """
    errors = []

    for param_name, values in param_grid.items():
        # Check if parameter is allowed
        if param_name not in ALLOWED_TUNING_PARAMS:
            errors.append(f"'{param_name}' is not an allowed tuning parameter")
            continue

        # Check if it's a list
        if not isinstance(values, list):
            errors.append(f"'{param_name}' values must be a list")
            continue

        # Check if list is not empty
        if len(values) == 0:
            errors.append(f"'{param_name}' values list cannot be empty")

    return len(errors) == 0, errors


def get_param_info() -> Dict[str, Any]:
    """Get information about allowed tuning parameters"""
    # Return a simple dict with the allowed parameters
    # Frontend will handle the UI details
    return {
        "allowed_params": list(ALLOWED_TUNING_PARAMS),
        "default_grid": DEFAULT_PARAM_GRID,
    }
