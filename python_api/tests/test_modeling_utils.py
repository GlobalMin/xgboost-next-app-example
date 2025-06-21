"""Tests for modeling utility functions"""

import pytest
import pandas as pd
import numpy as np
import os
from unittest.mock import patch, MagicMock
import xgboost as xgb

import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modeling_utils import (
    load_dataset,
    preprocess_data,
    create_train_test_split,
    generate_parameter_grid,
    calculate_feature_importance,
    calculate_lift_chart,
    cross_validate_params,
)


class TestLoadDataset:
    """Test dataset loading functionality"""

    def test_load_existing_dataset(self, temp_csv_file, monkeypatch):
        """Test loading an existing CSV file"""
        # Mock the UPLOAD_DIR to point to temp directory
        upload_dir = os.path.dirname(temp_csv_file)
        filename = os.path.basename(temp_csv_file)

        with patch("modeling_utils.UPLOAD_DIR", upload_dir):
            df = load_dataset(filename)

            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0
            assert "target" in df.columns

    def test_load_nonexistent_dataset(self):
        """Test loading a non-existent CSV file"""
        with pytest.raises(FileNotFoundError) as exc_info:
            load_dataset("nonexistent.csv")

        assert "Dataset file not found" in str(exc_info.value)


class TestPreprocessData:
    """Test data preprocessing functionality"""

    def test_preprocess_numeric_and_categorical(self, sample_dataframe):
        """Test preprocessing with both numeric and categorical features"""
        feature_cols = [
            "numeric_feature_1",
            "numeric_feature_2",
            "categorical_feature_1",
            "categorical_feature_2",
        ]
        target_col = "target"

        X, y, artifacts = preprocess_data(sample_dataframe, feature_cols, target_col)

        # Check output types
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert isinstance(artifacts, dict)

        # Check no missing values after preprocessing
        assert not X.isnull().any().any()

        # Check all values are numeric
        assert X.dtypes.apply(lambda x: np.issubdtype(x, np.number)).all()

        # Check artifacts structure
        assert "numeric_columns" in artifacts
        assert "categorical_columns" in artifacts
        assert "encoders" in artifacts
        assert "imputers" in artifacts

    def test_preprocess_with_missing_target(self, sample_dataframe):
        """Test that missing values in target raise an error"""
        # Add missing values to target
        sample_dataframe.loc[0:5, "target"] = np.nan

        with pytest.raises(ValueError) as exc_info:
            preprocess_data(sample_dataframe, ["numeric_feature_1"], "target")

        assert "Target column contains missing values" in str(exc_info.value)

    def test_preprocess_categorical_target(self):
        """Test preprocessing with categorical target"""
        df = pd.DataFrame(
            {"feature1": [1, 2, 3, 4, 5], "target": ["A", "B", "A", "B", "A"]}
        )

        X, y, artifacts = preprocess_data(df, ["feature1"], "target")

        # Target should be encoded to numeric
        assert y.dtype in [np.int32, np.int64]
        assert "_target" in artifacts["encoders"]

        # Check we can decode back
        le = artifacts["encoders"]["_target"]
        assert list(le.classes_) == ["A", "B"]

    def test_preprocess_only_numeric_features(self):
        """Test preprocessing with only numeric features"""
        df = pd.DataFrame(
            {
                "feature1": [1.0, 2.0, 3.0, np.nan, 5.0],
                "feature2": [10, 20, 30, 40, 50],
                "target": [0, 1, 0, 1, 0],
            }
        )

        X, y, artifacts = preprocess_data(df, ["feature1", "feature2"], "target")

        # Check imputation worked
        assert not X.isnull().any().any()
        assert X.loc[3, "feature1"] != np.nan  # Should be imputed with median


class TestCreateTrainTestSplit:
    """Test train-test split functionality"""

    def test_basic_split(self, sample_dataframe):
        """Test basic train-test split"""
        X = sample_dataframe.drop("target", axis=1)
        y = sample_dataframe["target"]

        X_train, X_test, y_train, y_test = create_train_test_split(X, y, test_size=0.2)

        # Check sizes
        assert len(X_train) == 80  # 80% of 100
        assert len(X_test) == 20  # 20% of 100
        assert len(y_train) == 80
        assert len(y_test) == 20

        # Check no overlap
        train_indices = set(X_train.index)
        test_indices = set(X_test.index)
        assert len(train_indices.intersection(test_indices)) == 0

    def test_stratified_split(self):
        """Test that split is stratified"""
        # Create imbalanced dataset
        df = pd.DataFrame(
            {
                "feature1": range(100),
                "target": [0] * 80 + [1] * 20,  # 80% class 0, 20% class 1
            }
        )

        X = df[["feature1"]]
        y = df["target"]

        X_train, X_test, y_train, y_test = create_train_test_split(X, y, test_size=0.2)

        # Check class distribution is maintained
        train_ratio = (y_train == 1).sum() / len(y_train)
        test_ratio = (y_test == 1).sum() / len(y_test)

        assert abs(train_ratio - 0.2) < 0.05  # Should be close to 20%
        assert abs(test_ratio - 0.2) < 0.05


class TestGenerateParameterGrid:
    """Test parameter grid generation"""

    def test_generate_single_param_grid(self):
        """Test grid generation with single parameter"""
        grid = {"max_depth": [3, 4, 5]}
        combinations = generate_parameter_grid(grid)

        assert len(combinations) == 3
        assert combinations[0] == {"max_depth": 3}
        assert combinations[1] == {"max_depth": 4}
        assert combinations[2] == {"max_depth": 5}

    def test_generate_multi_param_grid(self):
        """Test grid generation with multiple parameters"""
        grid = {"max_depth": [3, 4], "learning_rate": [0.1, 0.01], "subsample": [0.8]}
        combinations = generate_parameter_grid(grid)

        assert len(combinations) == 4  # 2 * 2 * 1

        # Check all combinations are present
        expected = [
            {"max_depth": 3, "learning_rate": 0.1, "subsample": 0.8},
            {"max_depth": 3, "learning_rate": 0.01, "subsample": 0.8},
            {"max_depth": 4, "learning_rate": 0.1, "subsample": 0.8},
            {"max_depth": 4, "learning_rate": 0.01, "subsample": 0.8},
        ]

        for exp in expected:
            assert exp in combinations

    def test_empty_grid(self):
        """Test empty parameter grid"""
        combinations = generate_parameter_grid({})
        assert len(combinations) == 1
        assert combinations[0] == {}


class TestCalculateFeatureImportance:
    """Test feature importance calculation"""

    def test_feature_importance_extraction(self):
        """Test extracting feature importance from model"""
        # Create a mock XGBoost model
        mock_model = MagicMock(spec=xgb.Booster)
        mock_model.get_score.return_value = {"f0": 100, "f1": 50, "f2": 25}

        feature_names = ["feature_a", "feature_b", "feature_c"]
        importance = calculate_feature_importance(mock_model, feature_names)

        assert len(importance) == 3
        assert importance["feature_a"] == 100.0
        assert importance["feature_b"] == 50.0
        assert importance["feature_c"] == 25.0

    def test_feature_importance_with_names(self):
        """Test when model uses actual feature names"""
        mock_model = MagicMock(spec=xgb.Booster)
        mock_model.get_score.side_effect = [
            {"feature_a": 100, "feature_b": 50},  # gain
            {},  # weight (fallback)
            {},  # cover (fallback)
        ]

        feature_names = ["feature_a", "feature_b", "feature_c"]
        importance = calculate_feature_importance(mock_model, feature_names)

        assert importance["feature_a"] == 100.0
        assert importance["feature_b"] == 50.0
        assert importance["feature_c"] == 0.0  # Not in importance dict

    def test_feature_importance_fallback(self):
        """Test fallback to different importance types"""
        mock_model = MagicMock(spec=xgb.Booster)
        mock_model.get_score.side_effect = [
            {},  # gain returns empty
            {"f0": 10, "f1": 20},  # weight returns values
            {},  # cover
        ]

        feature_names = ["feature_a", "feature_b"]
        importance = calculate_feature_importance(mock_model, feature_names)

        assert importance["feature_a"] == 10.0
        assert importance["feature_b"] == 20.0


class TestCalculateLiftChart:
    """Test lift chart calculation"""

    def test_basic_lift_chart(self):
        """Test lift chart calculation with perfect predictions"""
        # Create perfect predictions
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_pred = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

        lift_data = calculate_lift_chart(y_true, y_pred, n_bins=2)

        assert len(lift_data) == 2

        # First bin should have low predictions and actual rate of 0
        assert lift_data[0]["bin"] == 1
        assert lift_data[0]["actual_rate"] == 0.0
        assert lift_data[0]["count"] == 5

        # Second bin should have high predictions and actual rate of 1
        assert lift_data[1]["bin"] == 2
        assert lift_data[1]["actual_rate"] == 1.0
        assert lift_data[1]["count"] == 5

    def test_lift_chart_with_random_predictions(self):
        """Test lift chart with random predictions"""
        np.random.seed(42)
        n_samples = 100
        y_true = np.random.choice([0, 1], n_samples)
        y_pred = np.random.rand(n_samples)

        lift_data = calculate_lift_chart(y_true, y_pred, n_bins=5)

        assert len(lift_data) <= 5  # May be less due to duplicates

        # Check structure
        for bin_data in lift_data:
            assert "bin" in bin_data
            assert "avg_prediction" in bin_data
            assert "actual_rate" in bin_data
            assert "count" in bin_data
            assert bin_data["count"] > 0


class TestCrossValidateParams:
    """Test cross-validation functionality"""

    @patch("modeling_utils.xgb.cv")
    def test_cross_validate_success(self, mock_cv):
        """Test successful cross-validation"""
        # Mock CV results
        cv_results = pd.DataFrame(
            {
                "test-auc-mean": [0.7, 0.75, 0.8, 0.82, 0.83],
                "test-auc-std": [0.05, 0.04, 0.03, 0.03, 0.02],
            }
        )
        mock_cv.return_value = cv_results

        # Create mock DMatrix
        mock_dtrain = MagicMock(spec=xgb.DMatrix)
        params = {"max_depth": 3, "learning_rate": 0.1}

        mean_score, std_score, optimal_rounds = cross_validate_params(
            mock_dtrain, params, cv_folds=3, num_rounds=100, early_stopping_rounds=10
        )

        assert mean_score == 0.83  # Last value
        assert std_score == 0.02
        assert optimal_rounds == 5  # Length of results

    @patch("modeling_utils.xgb.cv")
    def test_cross_validate_empty_results(self, mock_cv):
        """Test cross-validation with empty results"""
        mock_cv.return_value = pd.DataFrame()  # Empty results

        mock_dtrain = MagicMock(spec=xgb.DMatrix)
        params = {"max_depth": 3}

        mean_score, std_score, optimal_rounds = cross_validate_params(
            mock_dtrain, params, cv_folds=3, num_rounds=100, early_stopping_rounds=10
        )

        assert mean_score == 0.5  # Default
        assert std_score == 0.0
        assert optimal_rounds == 100  # Default to num_rounds
