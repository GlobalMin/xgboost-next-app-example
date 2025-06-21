"""Tests for Pydantic models"""

import pytest
from pydantic import ValidationError
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import TrainRequest, ModelInfo


class TestTrainRequest:
    """Test TrainRequest model validation"""

    def test_valid_train_request(self, valid_train_request):
        """Test that valid train request is accepted"""
        request = TrainRequest(**valid_train_request)
        assert request.model_name == "test_model"
        assert request.csv_filename == "test.csv"
        assert request.target_column == "target"
        assert request.feature_columns == ["feature1", "feature2"]
        assert request.test_size == 0.2
        assert request.cv_folds == 3
        assert request.early_stopping_rounds == 50
        assert request.objective == "binary:logistic"
        assert request.custom_param_grid is None

    def test_required_fields(self):
        """Test that required fields are enforced"""
        # Missing required fields
        with pytest.raises(ValidationError) as exc_info:
            TrainRequest(
                model_name="test",
                csv_filename="test.csv",
                # Missing target_column and feature_columns
            )  # type: ignore

        errors = exc_info.value.errors()
        error_fields = [e["loc"][0] for e in errors]
        assert "target_column" in error_fields
        assert "feature_columns" in error_fields

    def test_default_values(self):
        """Test that default values are applied correctly"""
        request = TrainRequest(
            model_name="test",
            csv_filename="test.csv",
            target_column="target",
            feature_columns=["feature1"],
        )

        # Check defaults
        assert request.test_size == 0.2
        assert request.cv_folds == 3
        assert request.early_stopping_rounds == 50
        assert request.objective == "binary:logistic"
        assert request.custom_param_grid is None

    def test_custom_param_grid_validation(self):
        """Test custom parameter grid validation"""
        # Valid custom param grid
        valid_request = TrainRequest(
            model_name="test",
            csv_filename="test.csv",
            target_column="target",
            feature_columns=["feature1"],
            custom_param_grid={"max_depth": [3, 4, 5], "learning_rate": [0.01, 0.1]},
        )
        assert valid_request.custom_param_grid is not None

        # Invalid custom param grid
        with pytest.raises(ValidationError) as exc_info:
            TrainRequest(
                model_name="test",
                csv_filename="test.csv",
                target_column="target",
                feature_columns=["feature1"],
                custom_param_grid={"invalid_param": [1, 2, 3]},
            )

        error = exc_info.value.errors()[0]
        assert "Invalid parameter grid" in error["msg"]

    def test_empty_feature_columns(self):
        """Test that empty feature columns list is rejected"""
        with pytest.raises(ValidationError):
            TrainRequest(
                model_name="test",
                csv_filename="test.csv",
                target_column="target",
                feature_columns=[],  # Empty list should be invalid
            )

    def test_invalid_test_size(self):
        """Test test_size validation"""
        # Test size should be between 0 and 1
        with pytest.raises(ValidationError):
            TrainRequest(
                model_name="test",
                csv_filename="test.csv",
                target_column="target",
                feature_columns=["feature1"],
                test_size=1.5,  # Invalid
            )

        with pytest.raises(ValidationError):
            TrainRequest(
                model_name="test",
                csv_filename="test.csv",
                target_column="target",
                feature_columns=["feature1"],
                test_size=-0.1,  # Invalid
            )


class TestModelInfo:
    """Test ModelInfo model validation"""

    def test_valid_model_info(self):
        """Test that valid model info is accepted"""
        model_info = ModelInfo(
            id=1,
            name="test_model",
            created_at="2024-01-01 12:00:00",
            csv_filename="test.csv",
            target_column="target",
            feature_columns=["feature1", "feature2"],
            model_params={"max_depth": 5, "learning_rate": 0.1},
            auc_score=0.85,
            accuracy=0.82,
            feature_importance={"feature1": 0.6, "feature2": 0.4},
            confusion_matrix=[[50, 10], [5, 35]],
            status="completed",
        )

        assert model_info.id == 1
        assert model_info.name == "test_model"
        assert model_info.auc_score == 0.85
        assert model_info.accuracy == 0.82
        assert model_info.status == "completed"

    def test_optional_fields(self):
        """Test that optional fields can be None"""
        model_info = ModelInfo(
            id=1,
            name="test_model",
            created_at="2024-01-01 12:00:00",
            csv_filename="test.csv",
            target_column="target",
            feature_columns=["feature1"],
            model_params={},
            status="training",
            # All score fields are optional
        )

        assert model_info.auc_score is None
        assert model_info.accuracy is None
        assert model_info.feature_importance is None
        assert model_info.confusion_matrix is None

    def test_model_config_protected_namespaces(self):
        """Test that model_config is properly set"""
        # This should not raise an error about protected namespaces
        model = ModelInfo(
            id=1,
            name="test_model",
            created_at="2024-01-01",
            csv_filename="test.csv",
            target_column="target",
            feature_columns=["f1"],
            model_params={},
            status="completed",
        )

        # Check that we can access model_config
        assert hasattr(model, "model_config")
        assert model.model_config == {"protected_namespaces": ()}
