"""Tests for XGBoost parameter validation functions"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from xgb_params import (
    validate_param_name,
    validate_param_value,
    validate_param_grid,
    get_param_info,
    TUNABLE_PARAMS,
    DEFAULT_PARAM_GRID
)


class TestParamNameValidation:
    """Test parameter name validation"""
    
    def test_valid_param_names(self):
        """Test that valid parameter names are accepted"""
        valid_names = ["learning_rate", "max_depth", "subsample", "lambda", "eta"]
        for name in valid_names:
            assert validate_param_name(name) is True
    
    def test_invalid_param_names(self):
        """Test that invalid parameter names are rejected"""
        invalid_names = ["invalid_param", "not_a_param", "", "123", "max-depth"]
        for name in invalid_names:
            assert validate_param_name(name) is False
    
    def test_alias_param_names(self):
        """Test that parameter aliases are recognized"""
        # eta is alias for learning_rate
        assert validate_param_name("eta") is True
        assert validate_param_name("learning_rate") is True
        
        # reg_lambda is alias for lambda
        assert validate_param_name("reg_lambda") is True
        assert validate_param_name("lambda") is True


class TestParamValueValidation:
    """Test parameter value validation"""
    
    def test_learning_rate_validation(self):
        """Test learning_rate parameter validation"""
        # Valid values
        valid_values = [0.01, 0.1, 0.5, 0.999]
        for value in valid_values:
            is_valid, error = validate_param_value("learning_rate", value)
            assert is_valid is True
            assert error == ""
        
        # Invalid values
        invalid_cases = [
            (0, "must be > 0"),  # Exclusive lower bound
            (1.5, "must be > 0 and <= 1"),
            (-0.1, "must be > 0"),
            ("0.1", "must be of type float"),
        ]
        
        for value, expected_error_part in invalid_cases:
            is_valid, error = validate_param_value("learning_rate", value)
            assert is_valid is False
            assert expected_error_part in error
    
    def test_max_depth_validation(self):
        """Test max_depth parameter validation"""
        # Valid values
        valid_values = [0, 1, 5, 10, 100]
        for value in valid_values:
            is_valid, error = validate_param_value("max_depth", value)
            assert is_valid is True
        
        # Invalid values
        invalid_cases = [
            (-1, "must be >= 0"),
            (3.5, "must be of type int"),
            ("5", "must be of type int"),
        ]
        
        for value, expected_error_part in invalid_cases:
            is_valid, error = validate_param_value("max_depth", value)
            assert is_valid is False
            assert expected_error_part in error
    
    def test_tree_method_validation(self):
        """Test tree_method parameter validation"""
        # Valid values
        valid_values = ["auto", "exact", "approx", "hist", "gpu_hist"]
        for value in valid_values:
            is_valid, error = validate_param_value("tree_method", value)
            assert is_valid is True
        
        # Invalid values
        invalid_values = ["invalid", "gpu", 123]
        for value in invalid_values:
            is_valid, error = validate_param_value("tree_method", value)
            assert is_valid is False
    
    def test_float_params_accept_int(self):
        """Test that float parameters accept integer values"""
        float_params = ["learning_rate", "subsample", "lambda", "alpha"]
        
        for param in float_params:
            # Integer should be accepted for float params
            is_valid, error = validate_param_value(param, 1)
            assert is_valid is True or "must be > 0 and <= 1" in error  # Some have range restrictions


class TestParamGridValidation:
    """Test parameter grid validation"""
    
    def test_valid_param_grid(self):
        """Test that valid parameter grids are accepted"""
        valid_grid = {
            "max_depth": [3, 4, 5],
            "learning_rate": [0.01, 0.1],
            "subsample": [0.8, 0.9]
        }
        
        is_valid, errors = validate_param_grid(valid_grid)
        assert is_valid is True
        assert len(errors) == 0
    
    def test_empty_value_list(self):
        """Test that empty value lists are rejected"""
        invalid_grid = {
            "max_depth": [],
            "learning_rate": [0.1]
        }
        
        is_valid, errors = validate_param_grid(invalid_grid)
        assert is_valid is False
        assert any("cannot be empty" in error for error in errors)
    
    def test_non_list_values(self):
        """Test that non-list values are rejected"""
        invalid_grid = {
            "max_depth": 5,  # Should be a list
            "learning_rate": [0.1]
        }
        
        is_valid, errors = validate_param_grid(invalid_grid)
        assert is_valid is False
        assert any("must be a list" in error for error in errors)
    
    def test_non_tunable_params(self):
        """Test that non-tunable parameters are rejected"""
        invalid_grid = {
            "objective": ["binary:logistic", "reg:linear"],  # Not tunable
            "max_depth": [3, 4]
        }
        
        is_valid, errors = validate_param_grid(invalid_grid)
        assert is_valid is False
        assert any("not a tunable parameter" in error for error in errors)
    
    def test_invalid_param_values_in_grid(self):
        """Test that invalid parameter values in grid are caught"""
        invalid_grid = {
            "max_depth": [3, -1, 5],  # -1 is invalid
            "learning_rate": [0.1, 2.0]  # 2.0 is invalid
        }
        
        is_valid, errors = validate_param_grid(invalid_grid)
        assert is_valid is False
        assert len(errors) >= 2  # At least 2 errors
    
    def test_default_param_grid(self):
        """Test that the default parameter grid is valid"""
        is_valid, errors = validate_param_grid(DEFAULT_PARAM_GRID)
        assert is_valid is True
        assert len(errors) == 0


class TestParamInfo:
    """Test parameter information retrieval"""
    
    def test_get_param_info(self):
        """Test that param info returns expected structure"""
        param_info = get_param_info()
        
        # Check that we get info for primary tunable params
        expected_params = ["learning_rate", "max_depth", "subsample"]
        for param in expected_params:
            assert param in param_info
            assert "type" in param_info[param]
            assert "description" in param_info[param]
            assert "name" in param_info[param]
    
    def test_tunable_params_consistency(self):
        """Test that TUNABLE_PARAMS is consistent with param definitions"""
        for param in TUNABLE_PARAMS:
            # Either the param itself or its alias should be valid
            if not validate_param_name(param):
                # Check if it's an alias of something valid
                alias_found = False
                for p in ["learning_rate", "alpha", "lambda"]:
                    if param in ["eta", "reg_alpha", "reg_lambda"]:
                        alias_found = True
                        break
                assert alias_found, f"{param} in TUNABLE_PARAMS is not valid"