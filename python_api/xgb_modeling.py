"""Main XGBoost modeling workflow using modular components"""

import xgboost as xgb
from sklearn.metrics import roc_auc_score
from typing import Dict, Any


from models import TrainRequest
from mongo_utils import log_training_message, update_model_results, cleanup_failed_model
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


def train_xgboost_model(model_id: str, request: TrainRequest) -> Dict[str, Any]:
    """Main training workflow orchestrator"""
    try:
        # Step 1: Load and prepare data
        log_training_message(model_id, "Loading dataset...")
        df = load_dataset(request.csv_filename)

        log_training_message(model_id, "Preprocessing data...")
        X, y, label_encoders = preprocess_data(
            df, request.feature_columns, request.target_column
        )

        # Step 2: Create train-test split
        X_train, X_test, y_train, y_test = create_train_test_split(
            X, y, test_size=request.test_size
        )

        log_training_message(
            model_id,
            f"Dataset: {len(df)} rows, {len(X.columns)} features, "
            f"{len(X_train)}/{len(X_test)} train/test split",
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
            log_training_message(model_id, "Starting hyperparameter tuning...")

            param_grid = {
                "max_depth": [3, 4, 5],
                "learning_rate": [0.01, 0.05],
                "gamma": [0, 0.1, 0.3, 0.5],
                "subsample": [0.8],
                "colsample_bytree": [0.8],
            }

            best_params, cv_auc, best_n_estimators = tune_hyperparameters(
                dtrain,
                base_params,
                param_grid,
                request.cv_folds,
                request.early_stopping_rounds,
                model_id,
            )

            # Train final model with best parameters
            final_params = {**base_params, **best_params}
            log_training_message(
                model_id,
                f"Training final model with best params, {best_n_estimators} rounds...",
            )

            model = train_single_model(dtrain, final_params, best_n_estimators)

        else:
            log_training_message(model_id, "Training model without parameter tuning...")

            # Use default parameters
            default_params = {
                "max_depth": 6,
                "learning_rate": 0.1,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "gamma": 0,
                "min_child_weight": 1,
            }
            final_params = {**base_params, **default_params}

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

            best_params = default_params
            if hasattr(model, "best_iteration"):
                best_n_estimators = model.best_iteration

        # Step 5: Evaluate and save results
        log_training_message(model_id, "Evaluating model performance...")

        # Get predictions and calculate metrics
        y_pred_proba = model.predict(dtest)
        test_auc = roc_auc_score(y_test, y_pred_proba)

        # Calculate additional metrics
        feature_importance = calculate_feature_importance(
            model, request.feature_columns
        )
        lift_data = calculate_lift_chart(y_test.to_numpy(), y_pred_proba)

        # Save model
        _ = save_model(model, model_id)

        # Prepare final results
        results = prepare_results(
            float(test_auc),
            cv_auc,
            feature_importance,
            lift_data,
            best_params,
            best_n_estimators,
        )

        # Update model in database
        update_model_results(model_id, results)
        log_training_message(model_id, f"Training completed! Test AUC: {test_auc:.4f}")

        return {"model_id": model_id, "status": "completed", **results}

    except Exception as e:
        cleanup_failed_model(model_id, str(e))
        raise
