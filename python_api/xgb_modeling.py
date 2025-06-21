"""Main XGBoost modeling workflow using modular components"""

import xgboost as xgb
from sklearn.metrics import roc_auc_score
from typing import Dict, Any

from models import (
    TrainRequest,
    DataPreparationResult,
    TrainingResult,
)
from mongo_utils import (
    update_project_results,
    cleanup_failed_project,
    update_project_status,
    update_project_dataset_info,
    update_project_preprocessing,
    update_project_tuning_results,
)
from logging_config import get_logger
from xgb_params import DEFAULT_PARAM_GRID
from modeling_utils import (
    read_csv_file,
    split_train_test,
    fit_xgb_model,
    extract_feature_importance,
    compute_lift_chart_data,
    calculate_test_metrics,
    write_model_file,
    build_results_dict,
    execute_hyperparameter_search,
)
from preprocessing import preprocess_data, get_preprocessing_artifacts


# Get logger for this module
logger = get_logger(__name__)


def prepare_training_data(
    project_id: str, request: TrainRequest
) -> DataPreparationResult:
    """Phase 1: Load and prepare all training data"""
    update_project_status(project_id, "Loading and processing data")

    logger.info("Loading dataset...", extra={"project_id": project_id})
    df = read_csv_file(request.csv_filename)

    logger.info("Preprocessing data...", extra={"project_id": project_id})
    X, y, pipeline, preprocessing_info = preprocess_data(
        df, request.feature_columns, request.target_column
    )

    # Get preprocessing artifacts in backward-compatible format
    preprocessing_artifacts = get_preprocessing_artifacts(pipeline, preprocessing_info)

    # Add pipeline code to artifacts for storage
    preprocessing_artifacts["pipeline_code"] = preprocessing_info.get(
        "pipeline_code", ""
    )

    # Log preprocessing information
    n_numeric = len(preprocessing_artifacts.get("numeric_columns", []))
    n_categorical = len(preprocessing_artifacts.get("categorical_columns", []))
    logger.info(
        f"Feature types: {n_numeric} numeric, {n_categorical} categorical",
        extra={"project_id": project_id},
    )

    # Create train-test split
    X_train, X_test, y_train, y_test = split_train_test(
        X, y, test_size=request.test_size
    )

    logger.info(
        f"Dataset: {len(df)} rows, {len(X.columns)} features, "
        f"{len(X_train)}/{len(X_test)} train/test split",
        extra={"project_id": project_id},
    )

    # Convert to DMatrix format
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # Update dataset info in MongoDB
    dataset_updates = {
        "row_count": len(df),
        "column_count": len(df.columns),
        "column_types": {
            "numeric": preprocessing_artifacts.get("numeric_columns", []),
            "categorical": preprocessing_artifacts.get("categorical_columns", []),
            "datetime": preprocessing_artifacts.get("datetime_columns", []),
        },
        "missing_values": {
            col: int(df[col].isna().sum())
            for col in df.columns
            if df[col].isna().sum() > 0
        },
    }
    update_project_dataset_info(project_id, dataset_updates)

    # Update preprocessing info in MongoDB
    preprocessing_info = {
        "pipeline_definition": preprocessing_artifacts.get("pipeline_definition", {}),
        "feature_mappings": {
            k: v
            for k, v in preprocessing_artifacts.get("encoders", {}).items()
            if k not in ["categorical", "_target"]
        },
        "pipeline_code": preprocessing_artifacts.get("pipeline_code", ""),
    }
    update_project_preprocessing(project_id, preprocessing_info)

    # Prepare dataset info for later use
    dataset_info = {
        "total_rows": len(df),
        "n_features": len(X.columns),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "feature_columns": request.feature_columns,
    }

    return DataPreparationResult(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        dtrain=dtrain,
        dtest=dtest,
        preprocessing_artifacts=preprocessing_artifacts,
        dataset_info=dataset_info,
    )


def get_param_grid(request: TrainRequest) -> Dict[str, Any]:
    """Phase 2: Get parameter grid for tuning"""
    # Use custom parameter grid if provided, otherwise use default
    if request.custom_param_grid:
        return request.custom_param_grid
    else:
        return DEFAULT_PARAM_GRID


def train_and_tune_xgboost_model(
    dtrain: xgb.DMatrix, dtest: xgb.DMatrix, request: TrainRequest, project_id: str
) -> TrainingResult:
    """Phase 3: Train and tune the XGBoost model"""
    update_project_status(project_id, "Tuning xgboost for best hyperparameters")

    logger.info("Starting hyperparameter tuning...", extra={"project_id": project_id})

    # Get parameter grid
    param_grid = get_param_grid(request)

    # Log parameter grid info
    if isinstance(param_grid, dict) and len(param_grid) > 0:
        logger.info(
            f"Using parameter grid with {len(param_grid)} parameters",
            extra={"project_id": project_id},
        )
    else:
        logger.info("Using default parameter grid", extra={"project_id": project_id})

    # Tune hyperparameters
    best_params, cv_auc, best_n_estimators, cv_auc_std = execute_hyperparameter_search(
        dtrain,
        request.objective,
        param_grid,
        request.cv_folds,
        request.early_stopping_rounds,
        project_id,
    )

    # Update tuning results in MongoDB
    tuning_results = {
        "search_method": "grid_search",
        "param_grid": param_grid,
        "best_params": best_params,
        "best_score": cv_auc,
        "best_n_estimators": best_n_estimators,
    }
    update_project_tuning_results(project_id, tuning_results)

    # Train final model with best parameters
    update_project_status(project_id, "Finalizing model and calculating metrics")

    # Combine objective with tuned params
    final_params = {"objective": request.objective}
    final_params.update(best_params)

    logger.info(
        f"Training final model with best params, {best_n_estimators} rounds...",
        extra={"project_id": project_id},
    )

    model = fit_xgb_model(dtrain, final_params, best_n_estimators)

    return TrainingResult(
        model=model,
        best_params=best_params,
        cv_auc=cv_auc,
        cv_auc_std=cv_auc_std,
        best_n_estimators=best_n_estimators,
    )


def process_and_save_results(
    data_prep_result: DataPreparationResult,
    training_result: TrainingResult,
    project_id: str,
    request: TrainRequest,
) -> Dict[str, Any]:
    """Phase 4: Process results and save everything"""
    logger.info("Evaluating model performance...", extra={"project_id": project_id})

    # Get predictions and calculate metrics
    y_pred_proba = training_result.model.predict(data_prep_result.dtest)
    test_auc = roc_auc_score(data_prep_result.y_test, y_pred_proba)

    # Calculate additional metrics
    feature_importance = extract_feature_importance(
        training_result.model, request.feature_columns
    )
    lift_data = compute_lift_chart_data(
        data_prep_result.y_test.to_numpy(), y_pred_proba
    )
    test_metrics = calculate_test_metrics(
        data_prep_result.y_test.to_numpy(), y_pred_proba
    )

    # Save model
    _ = write_model_file(training_result.model, project_id)

    # Prepare final results
    final_params = {**training_result.best_params}
    results = build_results_dict(
        float(test_auc),
        training_result.cv_auc,
        feature_importance,
        lift_data,
        final_params,
        training_result.best_n_estimators,
        data_prep_result.preprocessing_artifacts,
        test_metrics,
        request.feature_columns,  # Pass feature names
        training_result.cv_auc_std,  # Pass CV std
    )

    # Update project in database
    update_project_results(project_id, results)
    logger.info(
        f"Training completed! Test AUC: {test_auc:.4f}",
        extra={"project_id": project_id},
    )

    return {"project_id": project_id, "status": "completed", **results}


def run_xgboost_training_flow(project_id: str, request: TrainRequest) -> Dict[str, Any]:
    """Main training workflow orchestrator - now modular and easy to follow"""
    try:
        # Phase 1: Prepare all training data
        data_prep_result = prepare_training_data(project_id, request)

        # Phase 2: Train and tune the XGBoost model
        training_result = train_and_tune_xgboost_model(
            data_prep_result.dtrain, data_prep_result.dtest, request, project_id
        )

        # Phase 4: Process results and save everything
        return process_and_save_results(
            data_prep_result, training_result, project_id, request
        )

    except Exception as e:
        cleanup_failed_project(project_id, str(e))
        raise
