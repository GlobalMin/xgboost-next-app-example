"""Preprocessing pipeline using scikit-learn's Pipeline and ColumnTransformer"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Dict, Any, Tuple, List, Optional


class DatetimeToString(BaseEstimator, TransformerMixin):
    """Convert datetime columns to strings"""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        datetime_columns = X_copy.select_dtypes(
            include=["datetime", "datetime64", "datetime64[ns]"]
        ).columns
        for col in datetime_columns:
            X_copy[col] = X_copy[col].astype(str)
        return X_copy


# Helper functions to break up the monolithic logic


def _detect_column_types(X: pd.DataFrame) -> Dict[str, List[str]]:
    """Detect and categorize column types in DataFrame"""
    datetime_columns = X.select_dtypes(
        include=["datetime", "datetime64", "datetime64[ns]"]
    ).columns.tolist()

    # Apply datetime conversion for type detection
    X_for_types = X.copy()
    for col in datetime_columns:
        X_for_types[col] = X_for_types[col].astype(str)

    numeric_columns = X_for_types.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = X_for_types.select_dtypes(include=["object"]).columns.tolist()

    return {
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "datetime_columns": datetime_columns,
        "all_columns": numeric_columns + categorical_columns,
    }


def _create_transformers(
    numeric_columns: List[str], categorical_columns: List[str]
) -> ColumnTransformer:
    """Create the column transformer with numeric and categorical pipelines"""
    transformers = []

    if numeric_columns:
        numeric_transformer = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="constant", fill_value=-9999))]
        )
        transformers.append(("num", numeric_transformer, numeric_columns))

    if categorical_columns:
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                (
                    "encoder",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                        max_categories=1000,  # type: ignore
                    ),
                ),
            ]
        )
        transformers.append(("cat", categorical_transformer, categorical_columns))

    return ColumnTransformer(
        transformers=transformers, remainder="passthrough", sparse_threshold=0
    )


def _build_column_info(column_types: Dict[str, List[str]]) -> Dict[str, Any]:
    """Build column information dictionary with pipeline code"""
    column_info: Dict[str, Any] = dict(column_types)  # Explicitly type as Any
    column_info["pipeline_code"] = generate_pipeline_code(
        column_types["numeric_columns"], column_types["categorical_columns"]
    )
    return column_info


def _process_target_variable(y: pd.Series) -> Tuple[pd.Series, Optional[LabelEncoder]]:
    """Handle target variable encoding if categorical"""
    if y.isnull().any():
        raise ValueError("Target column contains missing values")

    target_encoder = None
    if y.dtype == "object":
        target_encoder = LabelEncoder()
        y_encoded = target_encoder.fit_transform(y)
        y = pd.Series(np.asarray(y_encoded).tolist(), index=y.index, name=y.name)

    return y, target_encoder


def _extract_column_info_from_fitted_pipeline(
    pipeline: Pipeline,
) -> Dict[str, List[str]]:
    """Extract column information from a fitted pipeline"""
    preprocessor = pipeline.named_steps["preprocessor"]

    numeric_columns = []
    categorical_columns = []

    for name, transformer, columns in preprocessor.transformers_:
        if name == "num":
            numeric_columns = list(columns)
        elif name == "cat":
            categorical_columns = list(columns)

    return {
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "datetime_columns": [],  # Would need to be stored separately
        "all_columns": numeric_columns + categorical_columns,
    }


def _build_preprocessing_info(
    column_info: Dict[str, Any], target_encoder: Optional[LabelEncoder]
) -> Dict[str, Any]:
    """Build preprocessing info dictionary"""
    return {
        "column_info": column_info,
        "target_encoder_classes": target_encoder.classes_.tolist()
        if target_encoder
        else None,
        "feature_names": column_info["all_columns"],
        "pipeline_code": column_info.get("pipeline_code", ""),
    }


def _extract_encoder_artifacts(
    pipeline: Pipeline, column_info: Dict[str, Any]
) -> Dict[str, Any]:
    """Extract encoder artifacts from fitted pipeline"""
    artifacts = {"encoders": {}, "imputers": {}}

    if not column_info["categorical_columns"]:
        return artifacts

    preprocessor = pipeline.named_steps["preprocessor"]

    # Find categorical transformer
    cat_transformer = None
    for name, transformer, columns in preprocessor.transformers_:
        if name == "cat":
            cat_transformer = transformer
            break

    if cat_transformer:
        encoder = cat_transformer.named_steps["encoder"]
        artifacts["encoders"]["categorical"] = encoder

        # Store individual column categories
        for i, col in enumerate(column_info["categorical_columns"]):
            artifacts["encoders"][col] = {"categories": encoder.categories_[i].tolist()}

        artifacts["imputers"]["categorical"] = cat_transformer.named_steps["imputer"]

    return artifacts


# Main functions (now much cleaner)


def generate_pipeline_code(
    numeric_columns: List[str], categorical_columns: List[str]
) -> str:
    """Generate Python code to recreate the preprocessing pipeline"""
    code = f"""# Preprocessing pipeline definition
numeric_columns = {numeric_columns}
categorical_columns = {categorical_columns}

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value=-9999))
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_columns),
        ('cat', categorical_transformer, categorical_columns)
    ],
    remainder='passthrough',
    sparse_threshold=0
)

full_pipeline = Pipeline(steps=[
    ('datetime_converter', DatetimeToString()),
    ('preprocessor', preprocessor)
])"""
    return code


def create_preprocessing_pipeline(X: pd.DataFrame) -> Tuple[Pipeline, Dict[str, Any]]:
    """Create sklearn preprocessing pipeline for XGBoost"""
    column_types = _detect_column_types(X)
    preprocessor = _create_transformers(
        column_types["numeric_columns"], column_types["categorical_columns"]
    )

    full_pipeline = Pipeline(
        steps=[
            ("datetime_converter", DatetimeToString()),
            ("preprocessor", preprocessor),
        ]
    )

    column_info = _build_column_info(column_types)
    return full_pipeline, column_info


def preprocess_data(
    df: pd.DataFrame,
    feature_columns: List[str],
    target_column: str,
    pipeline: Optional[Pipeline] = None,
) -> Tuple[pd.DataFrame, pd.Series, Pipeline, Dict[str, Any]]:
    """Preprocess data using sklearn pipeline"""
    X = df[feature_columns].copy()
    y = df[target_column].copy()

    # Process target variable
    y, target_encoder = _process_target_variable(y)

    # Handle pipeline creation or reuse
    if pipeline is None:
        pipeline, column_info = create_preprocessing_pipeline(X)
        X_processed = pipeline.fit_transform(X)
    else:
        X_processed = pipeline.transform(X)
        column_info = _extract_column_info_from_fitted_pipeline(pipeline)

    # Convert to DataFrame with proper column names
    X_processed = pd.DataFrame(
        X_processed, index=X.index, columns=column_info["all_columns"]
    ).astype(np.float32)

    # Build preprocessing info
    preprocessing_info = _build_preprocessing_info(column_info, target_encoder)

    return X_processed, y, pipeline, preprocessing_info


def get_preprocessing_artifacts(
    pipeline: Optional[Pipeline], preprocessing_info: Dict[str, Any]
) -> Dict[str, Any]:
    """Extract preprocessing artifacts in a format compatible with existing code"""
    column_info = preprocessing_info["column_info"]

    artifacts = {
        "numeric_columns": column_info["numeric_columns"],
        "categorical_columns": column_info["categorical_columns"],
        "datetime_columns": column_info.get("datetime_columns", []),
        "encoders": {},
        "imputers": {},
        "pipeline_code": column_info.get("pipeline_code", ""),
    }

    # Build pipeline definition for new schema
    pipeline_definition = {
        "steps": [
            {
                "name": "datetime_converter",
                "type": "custom",
                "class": "DatetimeToString",
            },
            {
                "name": "column_transformer",
                "type": "sklearn.compose.ColumnTransformer",
                "transformers": [],
            },
        ]
    }

    # Add numeric transformer if needed
    if column_info["numeric_columns"]:
        pipeline_definition["steps"][1]["transformers"].append(
            {
                "name": "numeric",
                "transformer": "Pipeline",
                "columns": column_info["numeric_columns"],
                "steps": [
                    {
                        "name": "imputer",
                        "type": "SimpleImputer",
                        "params": {"strategy": "constant", "fill_value": -9999},
                    }
                ],
            }
        )

    # Add categorical transformer if needed
    if column_info["categorical_columns"]:
        pipeline_definition["steps"][1]["transformers"].append(
            {
                "name": "categorical",
                "transformer": "Pipeline",
                "columns": column_info["categorical_columns"],
                "steps": [
                    {
                        "name": "imputer",
                        "type": "SimpleImputer",
                        "params": {"strategy": "constant", "fill_value": "missing"},
                    },
                    {
                        "name": "encoder",
                        "type": "OrdinalEncoder",
                        "params": {
                            "handle_unknown": "use_encoded_value",
                            "unknown_value": -1,
                        },
                    },
                ],
            }
        )

    artifacts["pipeline_definition"] = pipeline_definition

    # Extract encoder artifacts if pipeline exists
    if pipeline and column_info["categorical_columns"]:
        encoder_artifacts = _extract_encoder_artifacts(pipeline, column_info)
        artifacts["encoders"].update(encoder_artifacts["encoders"])
        artifacts["imputers"].update(encoder_artifacts["imputers"])
    elif column_info["categorical_columns"]:
        # Store metadata about the imputer when no pipeline available
        artifacts["imputers"]["categorical"] = {
            "strategy": "constant",
            "fill_value": "missing",
        }

    # Add target encoder info if present
    if preprocessing_info.get("target_encoder_classes"):
        artifacts["encoders"]["_target"] = {
            "classes": preprocessing_info["target_encoder_classes"]
        }

    # Store numeric imputer info
    artifacts["imputers"]["numeric"] = "constant_-9999"

    return artifacts
