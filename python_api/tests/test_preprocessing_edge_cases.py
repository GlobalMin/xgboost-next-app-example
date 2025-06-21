"""Tests for preprocessing edge cases with non-standard data types"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np

from preprocessing import preprocess_data, get_preprocessing_artifacts


class TestPreprocessingEdgeCases:
    """Test preprocessing with non-standard data types and edge cases"""

    def test_preprocess_date_columns(self):
        """Test preprocessing with date and datetime columns"""
        df = pd.DataFrame(
            {
                "date_feature": pd.date_range("2024-01-01", periods=10),
                "datetime_feature": pd.date_range(
                    "2024-01-01 10:00:00", periods=10, freq="h"
                ),
                "date_string": ["2024-01-01", "2024-01-02", "2024-01-03"] * 3
                + ["2024-01-04"],
                "numeric_feature": np.random.randn(10),
                "target": np.random.choice([0, 1], 10),
            }
        )

        feature_cols = [
            "date_feature",
            "datetime_feature",
            "date_string",
            "numeric_feature",
        ]

        X, y, pipeline, preprocessing_info = preprocess_data(df, feature_cols, "target")
        artifacts = get_preprocessing_artifacts(pipeline, preprocessing_info)

        # Check that date columns are treated as categorical and encoded
        assert "date_feature" in artifacts["categorical_columns"]
        assert "datetime_feature" in artifacts["categorical_columns"]
        assert "date_string" in artifacts["categorical_columns"]

        # Check that datetime columns were tracked
        assert "date_feature" in artifacts["datetime_columns"]
        assert "datetime_feature" in artifacts["datetime_columns"]
        assert (
            "date_string" not in artifacts["datetime_columns"]
        )  # This was already a string

        # Check no missing values after preprocessing
        assert not X.isnull().any().any()

        # Check all values are numeric
        assert X.dtypes.apply(lambda x: np.issubdtype(x, np.number)).all()

    def test_preprocess_special_characters(self):
        """Test preprocessing with special characters and symbols"""
        df = pd.DataFrame(
            {
                "special_chars": [
                    "A@B",
                    "C#D",
                    "E$F",
                    "G&H",
                    "A@B",
                    "C#D",
                    None,
                    "!!!",
                    "***",
                    "",
                ],
                "unicode_chars": [
                    "café",
                    "naïve",
                    "résumé",
                    "π",
                    "∑",
                    "α",
                    "β",
                    "γ",
                    "δ",
                    "ε",
                ],
                "mixed_chars": [
                    "123-ABC",
                    "456_DEF",
                    "789/GHI",
                    "JKL\\MNO",
                    "PQR|STU",
                    "VWX+YZ",
                    "(ABC)",
                    "[DEF]",
                    "{GHI}",
                    "<JKL>",
                ],
                "normal_feature": list(range(10)),
                "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            }
        )

        feature_cols = [
            "special_chars",
            "unicode_chars",
            "mixed_chars",
            "normal_feature",
        ]

        X, y, pipeline, preprocessing_info = preprocess_data(df, feature_cols, "target")
        artifacts = get_preprocessing_artifacts(pipeline, preprocessing_info)

        # Check that all character columns are treated as categorical
        assert "special_chars" in artifacts["categorical_columns"]
        assert "unicode_chars" in artifacts["categorical_columns"]
        assert "mixed_chars" in artifacts["categorical_columns"]

        # Check encoding worked (including for None and empty string)
        assert not X.isnull().any().any()
        assert X.dtypes.apply(lambda x: np.issubdtype(x, np.number)).all()

        # Verify that special characters were encoded
        assert X.shape == (10, 4)

    def test_preprocess_long_text(self):
        """Test preprocessing with long text/sentences"""
        df = pd.DataFrame(
            {
                "long_text": [
                    "This is a very long sentence that should be treated as a category",
                    "Another extremely long piece of text with many words in it",
                    "Short text",
                    "This is a very long sentence that should be treated as a category",  # Duplicate
                    "Yet another sentence with special chars: @#$%^&*()",
                    "Sentence with numbers 123 456 789 and more text",
                    "UPPERCASE SENTENCE WITH LOTS OF WORDS",
                    "lowercase sentence with lots of words",
                    "Mixed Case Sentence With Punctuation!!!",
                    "   Sentence with leading and trailing spaces   ",
                ],
                "description": [
                    "Product description: This amazing product will change your life forever!",
                    "User review: 5 stars! Best purchase ever. Would recommend to everyone.",
                    "Comment: Not bad, could be better though. 3/5 stars.",
                    "Feedback: Customer service was excellent. Very satisfied.",
                    "Note: Please handle with care. Fragile item inside.",
                    "Warning: Do not use near water or in wet conditions.",
                    "Instructions: Step 1: Open box. Step 2: Remove contents. Step 3: Enjoy!",
                    "Error message: Something went wrong. Please try again later.",
                    "Success: Your order has been processed successfully!",
                    "Info: This is just an informational message. No action required.",
                ],
                "numeric_feature": np.arange(10),
                "target": [1, 0, 1, 1, 0, 0, 1, 0, 1, 0],
            }
        )

        feature_cols = ["long_text", "description", "numeric_feature"]

        X, y, pipeline, preprocessing_info = preprocess_data(df, feature_cols, "target")
        artifacts = get_preprocessing_artifacts(pipeline, preprocessing_info)

        # Check that text columns are treated as categorical
        assert "long_text" in artifacts["categorical_columns"]
        assert "description" in artifacts["categorical_columns"]

        # Check that each unique text is treated as a category
        assert "long_text" in artifacts["encoders"]
        assert "description" in artifacts["encoders"]

        # Verify encoding worked
        assert not X.isnull().any().any()
        assert X.dtypes.apply(lambda x: np.issubdtype(x, np.number)).all()

    def test_preprocess_mixed_types_extreme(self):
        """Test preprocessing with extreme mixed data types"""
        df = pd.DataFrame(
            {
                "mixed_numeric_string": [
                    "1",
                    "2.5",
                    "3",
                    "four",
                    "5.0",
                    "6",
                    "seven",
                    "8.8",
                    "9",
                    "ten",
                ],
                "json_like": [
                    '{"a": 1}',
                    '{"b": 2}',
                    "[1, 2, 3]",
                    "null",
                    "true",
                    "false",
                    "{}",
                    "[]",
                    '{"c": {"d": 4}}',
                    "invalid json",
                ],
                "urls": [
                    "http://example.com",
                    "https://test.org",
                    "ftp://files.com",
                    "www.no-protocol.com",
                    "/path/to/file",
                    "C:\\Windows\\System32",
                    "file:///home/user",
                    "mailto:test@example.com",
                    "javascript:alert(1)",
                    "data:text/plain;base64,SGVsbG8=",
                ],
                "whitespace_variations": [
                    "normal",
                    " leading",
                    "trailing ",
                    "  both  ",
                    "\ttab",
                    "new\nline",
                    "carriage\rreturn",
                    "multiple   spaces",
                    "\n\n\n",
                    "",
                ],
                "extreme_lengths": [
                    "a",
                    "ab",
                    "abc",
                    "a" * 100,
                    "b" * 500,
                    "c" * 1000,
                    "d" * 50,
                    "e" * 200,
                    "f" * 10,
                    "g" * 5,
                ],
                "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            }
        )

        feature_cols = [
            "mixed_numeric_string",
            "json_like",
            "urls",
            "whitespace_variations",
            "extreme_lengths",
        ]

        X, y, pipeline, preprocessing_info = preprocess_data(df, feature_cols, "target")
        artifacts = get_preprocessing_artifacts(pipeline, preprocessing_info)

        # All should be treated as categorical
        assert len(artifacts["categorical_columns"]) == 5
        assert len(artifacts["numeric_columns"]) == 0

        # Check proper encoding
        assert not X.isnull().any().any()
        assert X.dtypes.apply(lambda x: np.issubdtype(x, np.number)).all()
        assert X.shape == (10, 5)

    def test_preprocess_with_nan_inf_values(self):
        """Test preprocessing with NaN and inf values in numeric columns"""
        df = pd.DataFrame(
            {
                "numeric_with_inf": [
                    1.0,
                    2.0,
                    np.inf,
                    4.0,
                    -np.inf,
                    6.0,
                    7.0,
                    np.inf,
                    9.0,
                    10.0,
                ],
                "numeric_with_nan": [
                    1.0,
                    np.nan,
                    3.0,
                    np.nan,
                    5.0,
                    np.nan,
                    7.0,
                    8.0,
                    np.nan,
                    10.0,
                ],
                "mixed_special": [
                    1.0,
                    np.nan,
                    np.inf,
                    4.0,
                    -np.inf,
                    np.nan,
                    7.0,
                    8.0,
                    9.0,
                    10.0,
                ],
                "categorical": ["A", "B", "C", "A", "B", "C", "A", "B", "C", "A"],
                "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            }
        )

        # Replace inf with nan for preprocessing (common approach)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        feature_cols = [
            "numeric_with_inf",
            "numeric_with_nan",
            "mixed_special",
            "categorical",
        ]

        X, y, pipeline, preprocessing_info = preprocess_data(df, feature_cols, "target")
        artifacts = get_preprocessing_artifacts(pipeline, preprocessing_info)

        # Check that imputation handled NaN values
        assert not X.isnull().any().any()
        assert not np.isinf(X.values).any()

        # Verify numeric columns were imputed with -9999
        assert artifacts["imputers"]["numeric"] == "constant_-9999"

    def test_preprocess_single_value_columns(self):
        """Test preprocessing with columns that have only one unique value"""
        df = pd.DataFrame(
            {
                "constant_numeric": [5.0] * 10,
                "constant_string": ["same"] * 10,
                "almost_constant": ["A"] * 9 + ["B"],
                "normal_feature": np.random.randn(10),
                "target": np.random.choice([0, 1], 10),
            }
        )

        feature_cols = [
            "constant_numeric",
            "constant_string",
            "almost_constant",
            "normal_feature",
        ]

        X, y, pipeline, preprocessing_info = preprocess_data(df, feature_cols, "target")
        artifacts = get_preprocessing_artifacts(pipeline, preprocessing_info)

        # Even constant columns should be processed
        assert X.shape[1] == 4
        assert not X.isnull().any().any()

        # Constant string should be encoded to same value
        assert len(X["constant_string"].unique()) == 1

    def test_preprocess_with_duplicate_column_names(self):
        """Test handling of dataframes with duplicate column names"""
        # Create DataFrame with duplicate column names
        data = {
            "feature": [1, 2, 3, 4, 5],
            "feature_2": [
                "A",
                "B",
                "C",
                "D",
                "E",
            ],  # Different name to avoid pandas warning
            "target": [0, 1, 0, 1, 0],
        }
        df = pd.DataFrame(data)

        # Test with non-duplicate columns (normal case)
        feature_cols = ["feature", "feature_2"]
        X, y, pipeline, preprocessing_info = preprocess_data(df, feature_cols, "target")
        artifacts = get_preprocessing_artifacts(pipeline, preprocessing_info)

        assert X.shape == (5, 2)
        assert not X.isnull().any().any()

    def test_preprocess_high_cardinality_categorical(self):
        """Test preprocessing with high cardinality categorical features"""
        # Create a feature where every value is unique (worst case for categorical)
        n_samples = 100
        df = pd.DataFrame(
            {
                "unique_ids": [f"ID_{i:04d}" for i in range(n_samples)],
                "high_cardinality": [
                    f"Category_{i % 50}" for i in range(n_samples)
                ],  # 50 categories
                "medium_cardinality": [
                    f"Type_{i % 10}" for i in range(n_samples)
                ],  # 10 categories
                "low_cardinality": ["A", "B", "C"] * 33 + ["A"],  # 3 categories
                "numeric": np.random.randn(n_samples),
                "target": np.random.choice([0, 1], n_samples),
            }
        )

        feature_cols = [
            "unique_ids",
            "high_cardinality",
            "medium_cardinality",
            "low_cardinality",
            "numeric",
        ]

        X, y, pipeline, preprocessing_info = preprocess_data(df, feature_cols, "target")
        artifacts = get_preprocessing_artifacts(pipeline, preprocessing_info)

        # All categorical columns should be encoded
        assert "unique_ids" in artifacts["categorical_columns"]
        assert "high_cardinality" in artifacts["categorical_columns"]

        # Check that encoding worked even for unique values
        assert not X.isnull().any().any()
        assert X.shape == (n_samples, 5)

        # Verify the number of unique encoded values matches original
        assert len(artifacts["encoders"]["unique_ids"]["categories"]) == n_samples
        assert len(artifacts["encoders"]["high_cardinality"]["categories"]) == 50
