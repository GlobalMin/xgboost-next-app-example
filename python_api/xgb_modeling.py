"""Main XGBoost modeling workflow using modular components"""

import xgboost as xgb
from sklearn.metrics import roc_auc_score
from typing import Dict, Any


from python_api.models import TrainRequest
from mongo_utils import (
    update_project_results,
    cleanup_failed_project,
    update_project_status,
)
from logging_config import get_logger
from xgb_params import DEFAULT_PARAM_GRID
from modeling_utils import (
    load_dataset,
    preprocess_data,
    create_train_test_split,
    train_single_model,
    cross_validate_params,
    calculate_feature_importance,
    calculate_lift_chart,
    save_model,
    prepare_results,
    tune_hyperparameters,
)


# Get logger for this module
logger = get_logger(__name__)


def train_xgboost_model(project_id: str, request: TrainRequest) -> Dict[str, Any]:
    """Main training workflow orchestrator"""
    try:
        # Step 1: Load and prepare data
        update_project_status(project_id, "Loading and processing data")
        logger.debug(
            "Updated status to 'Loading and processing data'",
            extra={"project_id": project_id},
        )
        logger.info("Loading dataset...", extra={"project_id": project_id})
        df = load_dataset(request.csv_filename)

        logger.info("Preprocessing data...", extra={"project_id": project_id})
        X, y, preprocessing_artifacts = preprocess_data(
            df, request.feature_columns, request.target_column
        )

        # Log preprocessing information
        n_numeric = len(preprocessing_artifacts.get("numeric_columns", []))
        n_categorical = len(preprocessing_artifacts.get("categorical_columns", []))
        logger.info(
            f"Feature types: {n_numeric} numeric, {n_categorical} categorical",
            extra={"project_id": project_id},
        )

        # Step 2: Create train-test split
        X_train, X_test, y_train, y_test = create_train_test_split(
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

        # Step 3: Set up base parameters
        base_params = {
            "objective": request.objective,
            "eval_metric": "auc",
            "seed": 42,
            "nthread": -1,
        }

        # Step 4: Train model (with or without hyperparameter tuning)
        if request.tune_parameters:
            update_project_status(project_id, "Tuning xgboost for best hyperparameters")
            logger.debug(
                "Updated status to 'Tuning xgboost for best hyperparameters'",
                extra={"project_id": project_id},
            )
            logger.info(
                "Starting hyperparameter tuning...", extra={"project_id": project_id}
            )

            # Use custom parameter grid if provided, otherwise use default
            if request.custom_param_grid:
                param_grid = request.custom_param_grid
                logger.info(
                    f"Using custom parameter grid with {len(param_grid)} parameters",
                    extra={"project_id": project_id},
                )
            else:
                param_grid = DEFAULT_PARAM_GRID
                logger.info(
                    "Using default parameter grid", extra={"project_id": project_id}
                )

            best_params, cv_auc, best_n_estimators = tune_hyperparameters(
                dtrain,
                base_params,
                param_grid,
                request.cv_folds,
                request.early_stopping_rounds,
                project_id,
            )

            # Train final model with best parameters
            update_project_status(
                project_id, "Finalizing model and calculating metrics"
            )
            logger.debug(
                "Updated status to 'Finalizing model and calculating metrics'",
                extra={"project_id": project_id},
            )

            final_params = {**base_params, **best_params}
            logger.info(
                f"Training final model with best params, {best_n_estimators} rounds...",
                extra={"project_id": project_id},
            )

            model = train_single_model(dtrain, final_params, best_n_estimators)

        else:
            update_project_status(project_id, "Training xgboost model")
            logger.debug(
                "Updated status to 'Training xgboost model'",
                extra={"project_id": project_id},
            )
            # No hyperparameter tuning, use base parameters
            logger.info(
                "Training model without parameter tuning...",
                extra={"project_id": project_id},
            )

            # Use custom parameters (only override what's different from XGBoost defaults)
            custom_params = {
                "learning_rate": 0.1,  # XGBoost default is 0.3, but 0.1 is often better
            }
            final_params = {**base_params, **custom_params}

            # Run cross-validation
            cv_auc, _, best_n_estimators = cross_validate_params(
                dtrain,
                final_params,
                request.cv_folds,
                200,
                request.early_stopping_rounds,
            )

            # Train final model
            model = train_single_model(
                dtrain,
                final_params,
                best_n_estimators,
                evals=[(dtest, "eval")],
                early_stopping_rounds=request.early_stopping_rounds,
            )

            best_params = custom_params
            if hasattr(model, "best_iteration"):
                best_n_estimators = model.best_iteration

        # Step 5: Evaluate and save results
        logger.info("Evaluating model performance...", extra={"project_id": project_id})

        # Get predictions and calculate metrics
        y_pred_proba = model.predict(dtest)
        test_auc = roc_auc_score(y_test, y_pred_proba)

        # Calculate additional metrics
        feature_importance = calculate_feature_importance(
            model, request.feature_columns
        )
        lift_data = calculate_lift_chart(y_test.to_numpy(), y_pred_proba)

        # Save model
        _ = save_model(model, project_id)

        # Prepare final results
        results = prepare_results(
            float(test_auc),
            cv_auc,
            feature_importance,
            lift_data,
            final_params,
            best_n_estimators,
            preprocessing_artifacts,
        )

        # Update project in database
        update_project_results(project_id, results)
        logger.info(
            f"Training completed! Test AUC: {test_auc:.4f}",
            extra={"project_id": project_id},
        )

        return {"project_id": project_id, "status": "completed", **results}

    except Exception as e:
        cleanup_failed_project(project_id, str(e))
        raise
