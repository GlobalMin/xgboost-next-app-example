"""XGBoost parameter definitions and validation"""

from typing import Dict, List, Any

# Valid XGBoost parameters with their types and constraints
# Based on XGBoost documentation: https://xgboost.readthedocs.io/en/latest/parameter.html
VALID_XGBOOST_PARAMS = {
    # Tree Booster Parameters
    "eta": {
        "type": float,
        "range": (0, 1),
        "alias": ["learning_rate"],
        "description": "Step size shrinkage",
    },
    "learning_rate": {
        "type": float,
        "range": (0, 1),
        "alias": ["eta"],
        "description": "Step size shrinkage",
    },
    "gamma": {
        "type": float,
        "range": [0, float("inf")],
        "description": "Minimum loss reduction",
    },
    "max_depth": {
        "type": int,
        "range": [0, float("inf")],
        "description": "Maximum tree depth",
    },
    "min_child_weight": {
        "type": float,
        "range": [0, float("inf")],
        "description": "Minimum sum of instance weight",
    },
    "max_delta_step": {
        "type": float,
        "range": [0, float("inf")],
        "description": "Maximum delta step",
    },
    "subsample": {
        "type": float,
        "range": (0, 1),
        "description": "Subsample ratio of training instances",
    },
    "colsample_bytree": {
        "type": float,
        "range": (0, 1),
        "description": "Subsample ratio of columns when constructing each tree",
    },
    "colsample_bylevel": {
        "type": float,
        "range": (0, 1),
        "description": "Subsample ratio of columns for each level",
    },
    "colsample_bynode": {
        "type": float,
        "range": (0, 1),
        "description": "Subsample ratio of columns for each node",
    },
    "lambda": {
        "type": float,
        "range": [0, float("inf")],
        "alias": ["reg_lambda"],
        "description": "L2 regularization term",
    },
    "reg_lambda": {
        "type": float,
        "range": [0, float("inf")],
        "alias": ["lambda"],
        "description": "L2 regularization term",
    },
    "alpha": {
        "type": float,
        "range": [0, float("inf")],
        "alias": ["reg_alpha"],
        "description": "L1 regularization term",
    },
    "reg_alpha": {
        "type": float,
        "range": [0, float("inf")],
        "alias": ["alpha"],
        "description": "L1 regularization term",
    },
    "tree_method": {
        "type": str,
        "values": ["auto", "exact", "approx", "hist", "gpu_hist"],
        "description": "Tree construction algorithm",
    },
    "scale_pos_weight": {
        "type": float,
        "range": (0, float("inf")),
        "description": "Balance of positive and negative weights",
    },
    "max_leaves": {
        "type": int,
        "range": [0, float("inf")],
        "description": "Maximum number of leaves",
    },
    "grow_policy": {
        "type": str,
        "values": ["depthwise", "lossguide"],
        "description": "Tree growing policy",
    },
    # Learning Task Parameters (these shouldn't be in param_grid as they're not tuned)
    "objective": {
        "type": str,
        "values": [
            "binary:logistic",
            "binary:logitraw",
            "binary:hinge",
            "multi:softmax",
            "multi:softprob",
            "reg:linear",
            "reg:logistic",
            "reg:gamma",
            "reg:tweedie",
        ],
        "description": "Learning objective",
    },
    "eval_metric": {
        "type": str,
        "values": [
            "rmse",
            "mae",
            "logloss",
            "error",
            "merror",
            "mlogloss",
            "auc",
            "aucpr",
            "map",
        ],
        "description": "Evaluation metric",
    },
    "seed": {"type": int, "range": [0, float("inf")], "description": "Random seed"},
    "nthread": {
        "type": int,
        "range": [-1, float("inf")],
        "description": "Number of threads",
    },
    # Other parameters
    "verbosity": {
        "type": int,
        "values": [0, 1, 2, 3],
        "description": "Verbosity level",
    },
    "disable_default_eval_metric": {
        "type": int,
        "values": [0, 1],
        "description": "Disable default metric",
    },
    "num_parallel_tree": {
        "type": int,
        "range": [1, float("inf")],
        "description": "Number of parallel trees",
    },
}

# Parameters that are typically tuned in hyperparameter search
# Restricted to only the most important parameters
TUNABLE_PARAMS = {
    "learning_rate",
    "eta",  # eta is alias for learning_rate
    "alpha",
    "reg_alpha",  # reg_alpha is alias for alpha
    "lambda",
    "reg_lambda",  # reg_lambda is alias for lambda
    "max_depth",
    "min_child_weight",
    "subsample",
    "colsample_bytree",
}

# Default parameter grid for hyperparameter tuning
# Focused on the most important parameters to keep combinations reasonable (~24 total)
DEFAULT_PARAM_GRID = {
    "max_depth": [3, 4, 5],  # 3 values
    "learning_rate": [0.01, 0.1],  # 2 values
    "lambda": [0, 0.5, 1.0],  # 3 values
    "subsample": [0.8],  # 1 value
    "colsample_bytree": [0.8],  # 1 value
    # Total: 3 × 2 × 3 × 1 × 1 = 18 combinations
}


def validate_param_name(param_name: str) -> bool:
    """Check if a parameter name is valid for XGBoost"""
    # Check direct match or alias
    if param_name in VALID_XGBOOST_PARAMS:
        return True

    # Check if it's an alias
    for param, info in VALID_XGBOOST_PARAMS.items():
        if "alias" in info and param_name in info["alias"]:
            return True

    return False


def validate_param_value(param_name: str, value: Any) -> tuple[bool, str]:
    """
    Validate a parameter value for XGBoost
    Returns (is_valid, error_message)
    """
    if not validate_param_name(param_name):
        return False, f"'{param_name}' is not a valid XGBoost parameter"

    # Get parameter info (handle aliases)
    param_info = None

    if param_name in VALID_XGBOOST_PARAMS:
        param_info = VALID_XGBOOST_PARAMS[param_name]
    else:
        # Check aliases
        for param, info in VALID_XGBOOST_PARAMS.items():
            if "alias" in info and param_name in info["alias"]:
                param_info = info
                break

    if not param_info:
        return False, f"Could not find parameter info for '{param_name}'"

    # Check type
    expected_type = param_info["type"]
    # Allow int for float parameters
    if expected_type is float and isinstance(value, (int, float)):
        pass  # Both int and float are acceptable for float parameters
    elif not isinstance(value, expected_type):
        return (
            False,
            f"'{param_name}' must be of type {expected_type.__name__}, got {type(value).__name__}",
        )

    # Check specific values if defined
    if "values" in param_info:
        if value not in param_info["values"]:
            return (
                False,
                f"'{param_name}' must be one of {param_info['values']}, got '{value}'",
            )

    # Check range if defined
    if "range" in param_info:
        min_val, max_val = param_info["range"]
        if isinstance(param_info["range"], list):  # [min, max] inclusive on both ends
            if value < min_val or (max_val != float("inf") and value > max_val):
                return (
                    False,
                    f"'{param_name}' must be >= {min_val} and <= {max_val}, got {value}",
                )
        else:  # (min, max) tuple notation - exclusive min, inclusive max
            if value <= min_val or (max_val != float("inf") and value > max_val):
                return (
                    False,
                    f"'{param_name}' must be > {min_val} and <= {max_val}, got {value}",
                )

    return True, ""


def validate_param_grid(param_grid: Dict[str, List[Any]]) -> tuple[bool, List[str]]:
    """
    Validate a complete parameter grid
    Returns (is_valid, list_of_errors)
    """
    errors = []

    for param_name, values in param_grid.items():
        # Check if parameter is tunable
        if param_name not in TUNABLE_PARAMS:
            # Check if it's an alias of a tunable parameter
            is_tunable_alias = False
            for tunable_param in TUNABLE_PARAMS:
                if tunable_param in VALID_XGBOOST_PARAMS:
                    param_info = VALID_XGBOOST_PARAMS[tunable_param]
                    if "alias" in param_info and param_name in param_info["alias"]:
                        is_tunable_alias = True
                        break

            if not is_tunable_alias:
                errors.append(f"'{param_name}' is not a tunable parameter")
                continue

        # Check if it's a list
        if not isinstance(values, list):
            errors.append(
                f"'{param_name}' values must be a list, got {type(values).__name__}"
            )
            continue

        # Check if list is not empty
        if len(values) == 0:
            errors.append(f"'{param_name}' values list cannot be empty")
            continue

        # Validate each value
        for value in values:
            is_valid, error_msg = validate_param_value(param_name, value)
            if not is_valid:
                errors.append(error_msg)

    return len(errors) == 0, errors


def get_param_info() -> Dict[str, Any]:
    """Get information about all tunable parameters"""
    param_info = {}
    # Only include the primary parameter names (not aliases)
    primary_params = [
        "learning_rate",
        "alpha",
        "lambda",
        "max_depth",
        "min_child_weight",
        "subsample",
        "colsample_bytree",
    ]

    for param_name in primary_params:
        if param_name in VALID_XGBOOST_PARAMS:
            info = VALID_XGBOOST_PARAMS[param_name].copy()
            info["name"] = param_name
            param_info[param_name] = info
    return param_info
