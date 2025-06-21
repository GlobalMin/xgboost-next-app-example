"""Main XGBoost modeling workflow using modular components"""

import xgboost as xgb
from sklearn.metrics import roc_auc_score
from typing import Dict, Any

from models import (
    TrainRequest,
    DataPreparationResult,
    ModelConfiguration,
    TrainingResult,
)
from mongo_utils import (
    update_project_results,
    cleanup_failed_project,
    update_project_status,
)
from logging_config import get_logger
from xgb_params import DEFAULT_PARAM_GRID
from modeling_utils import (
    read_csv_file,
    process_raw_features,
    split_train_test,
    fit_xgb_model,
    extract_feature_importance,
    compute_lift_chart_data,
    write_model_file,
    build_results_dict,
    execute_hyperparameter_search,
)


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
    X, y, preprocessing_artifacts = process_raw_features(
        df, request.feature_columns, request.target_column
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


def setup_model_params(request: TrainRequest) -> ModelConfiguration:
    """Phase 2: Setup model parameters and configuration"""
    # Set up base parameters
    base_params = {
        "objective": request.objective,
        "eval_metric": "auc",
        "seed": 42,
        "nthread": -1,
    }

    # Use custom parameter grid if provided, otherwise use default
    if request.custom_param_grid:
        param_grid = request.custom_param_grid
    else:
        param_grid = DEFAULT_PARAM_GRID

    return ModelConfiguration(
        base_params=base_params,
        param_grid=param_grid,
        cv_folds=request.cv_folds,
        early_stopping_rounds=request.early_stopping_rounds,
    )


def train_and_tune_xgboost_model(
    dtrain: xgb.DMatrix, dtest: xgb.DMatrix, config: ModelConfiguration, project_id: str
) -> TrainingResult:
    """Phase 3: Train and tune the XGBoost model"""
    update_project_status(project_id, "Tuning xgboost for best hyperparameters")

    logger.info("Starting hyperparameter tuning...", extra={"project_id": project_id})

    # Log parameter grid info
    if isinstance(config.param_grid, dict) and len(config.param_grid) > 0:
        logger.info(
            f"Using parameter grid with {len(config.param_grid)} parameters",
            extra={"project_id": project_id},
        )
    else:
        logger.info("Using default parameter grid", extra={"project_id": project_id})

    # Tune hyperparameters
    best_params, cv_auc, best_n_estimators = execute_hyperparameter_search(
        dtrain,
        config.base_params,
        config.param_grid,
        config.cv_folds,
        config.early_stopping_rounds,
        project_id,
    )

    # Train final model with best parameters
    update_project_status(project_id, "Finalizing model and calculating metrics")

    final_params = {**config.base_params, **best_params}
    logger.info(
        f"Training final model with best params, {best_n_estimators} rounds...",
        extra={"project_id": project_id},
    )

    model = fit_xgb_model(dtrain, final_params, best_n_estimators)

    return TrainingResult(
        model=model,
        best_params=best_params,
        cv_auc=cv_auc,
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
    lift_data = compute_lift_chart_data(data_prep_result.y_test.to_numpy(), y_pred_proba)

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

        # Phase 2: Setup model parameters and configuration
        model_config = setup_model_params(request)

        # Phase 3: Train and tune the XGBoost model
        training_result = train_and_tune_xgboost_model(
            data_prep_result.dtrain, data_prep_result.dtest, model_config, project_id
        )

        # Phase 4: Process results and save everything
        return process_and_save_results(
            data_prep_result, training_result, project_id, request
        )

    except Exception as e:
        cleanup_failed_project(project_id, str(e))
        raise
