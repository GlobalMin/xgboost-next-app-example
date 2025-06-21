"""Pytest configuration and shared fixtures"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing"""
    np.random.seed(42)
    n_samples = 100

    data = {
        "numeric_feature_1": np.random.randn(n_samples),
        "numeric_feature_2": np.random.randint(0, 100, n_samples),
        "categorical_feature_1": np.random.choice(["A", "B", "C"], n_samples),
        "categorical_feature_2": np.random.choice(["X", "Y", "Z"], n_samples),
        "target": np.random.choice([0, 1], n_samples),
    }

    df = pd.DataFrame(data)

    # Add some missing values
    df.loc[5:10, "numeric_feature_1"] = np.nan
    df.loc[15:20, "categorical_feature_1"] = np.nan

    return df


@pytest.fixture
def temp_csv_file(sample_dataframe):
    """Create a temporary CSV file with sample data"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        sample_dataframe.to_csv(f, index=False)
        temp_path = f.name

    yield temp_path

    # Cleanup
    os.unlink(temp_path)


@pytest.fixture
def valid_train_request():
    """Create a valid TrainRequest dictionary"""
    return {
        "model_name": "test_model",
        "csv_filename": "test.csv",
        "target_column": "target",
        "feature_columns": ["feature1", "feature2"],
        "test_size": 0.2,
        "cv_folds": 3,
        "early_stopping_rounds": 50,
        "objective": "binary:logistic",
    }


@pytest.fixture
def mock_upload_dir(tmp_path):
    """Create a temporary upload directory"""
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    return str(upload_dir)
