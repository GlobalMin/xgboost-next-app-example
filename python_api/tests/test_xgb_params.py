"""Tests for simplified XGBoost parameter validation functions"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from xgb_params import (
    validate_param_grid,
    get_param_info,
    ALLOWED_TUNING_PARAMS,
    DEFAULT_PARAM_GRID,
)


class TestParamGridValidation:
    """Test parameter grid validation"""

    def test_valid_param_grid(self):
        """Test that valid parameter grids are accepted"""
        valid_grid = {
            "max_depth": [3, 4, 5],
            "learning_rate": [0.01, 0.1],
            "subsample": [0.8, 0.9],
        }

        is_valid, errors = validate_param_grid(valid_grid)
        assert is_valid is True
        assert len(errors) == 0

    def test_empty_value_list(self):
        """Test that empty value lists are rejected"""
        invalid_grid = {"max_depth": [], "learning_rate": [0.1]}

        is_valid, errors = validate_param_grid(invalid_grid)
        assert is_valid is False
        assert any("cannot be empty" in error for error in errors)

    def test_non_list_values(self):
        """Test that non-list values are rejected"""
        invalid_grid = {
            "max_depth": 5,  # Should be a list
            "learning_rate": [0.1],
        }

        is_valid, errors = validate_param_grid(invalid_grid)
        assert is_valid is False
        assert any("must be a list" in error for error in errors)

    def test_non_allowed_params(self):
        """Test that non-allowed parameters are rejected"""
        invalid_grid = {
            "objective": ["binary:logistic", "reg:linear"],  # Not allowed
            "max_depth": [3, 4],
        }

        is_valid, errors = validate_param_grid(invalid_grid)
        assert is_valid is False
        assert any("not an allowed tuning parameter" in error for error in errors)

    def test_default_param_grid(self):
        """Test that the default parameter grid is valid"""
        is_valid, errors = validate_param_grid(DEFAULT_PARAM_GRID)
        assert is_valid is True
        assert len(errors) == 0

    def test_aliases_accepted(self):
        """Test that parameter aliases are accepted"""
        valid_grid = {
            "eta": [0.01, 0.1],  # alias for learning_rate
            "reg_lambda": [0, 1],  # alias for lambda
            "reg_alpha": [0, 0.5],  # alias for alpha
        }

        is_valid, errors = validate_param_grid(valid_grid)
        assert is_valid is True
        assert len(errors) == 0


class TestParamInfo:
    """Test parameter information retrieval"""

    def test_get_param_info(self):
        """Test that param info returns expected structure"""
        param_info = get_param_info()

        # Check structure
        assert "allowed_params" in param_info
        assert "default_grid" in param_info

        # Check allowed params is a list
        assert isinstance(param_info["allowed_params"], list)
        assert len(param_info["allowed_params"]) > 0

        # Check default grid
        assert param_info["default_grid"] == DEFAULT_PARAM_GRID

    def test_allowed_params_consistency(self):
        """Test that all allowed params are in the set"""
        param_info = get_param_info()
        for param in param_info["allowed_params"]:
            assert param in ALLOWED_TUNING_PARAMS
