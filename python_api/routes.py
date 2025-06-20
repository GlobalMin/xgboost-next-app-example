from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.feature_selection import f_classif, f_regression
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from pydantic import BaseModel
from typing import List


from xgb_modeling import train_xgboost_model
from models import TrainRequest
from config import UPLOAD_DIR
from mongo_utils import create_project, get_project, get_projects, get_training_logs
from xgb_params import DEFAULT_PARAM_GRID, get_param_info
from code_generator import generate_training_code

# Create router instance
router = APIRouter(prefix="/api")


class FeatureSignalRequest(BaseModel):
    csv_filename: str
    target_column: str
    feature_columns: List[str]


@router.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    """Upload CSV file and return preview data"""
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    try:
        df = pd.read_csv(file_path)
        columns = df.columns.tolist()
        shape = df.shape
        dtypes = {col: str(df[col].dtype) for col in columns}

        # Prepare preview data
        preview_df = df.head(10).copy()
        preview_df = preview_df.fillna("")

        # Handle numeric values for JSON serialization
        for col in preview_df.columns:
            if preview_df[col].dtype in ["float64", "float32"]:
                preview_df[col] = preview_df[col].apply(
                    lambda x: x if pd.notna(x) and np.isfinite(x) else None
                )

        return {
            "filename": file.filename,
            "columns": columns,
            "shape": shape,
            "dtypes": dtypes,
            "preview": preview_df.to_dict(orient="records"),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading CSV: {str(e)}")


def run_training_task(project_id: str, request: TrainRequest):
    """Run training as a background task"""
    try:
        train_xgboost_model(project_id, request)
    except Exception as e:
        print(f"Training failed for project {project_id}: {str(e)}")
        # Update status to failed
        from mongo_utils import update_project_status

        update_project_status(project_id, "failed")


@router.post("/train")
async def train_project(request: TrainRequest, background_tasks: BackgroundTasks):
    """Start project training with provided configuration"""
    # Create project document in MongoDB
    project_data = {
        "name": request.model_name,
        "csv_filename": request.csv_filename,
        "target_column": request.target_column,
        "feature_columns": request.feature_columns,
        "test_size": request.test_size,
        "cv_folds": request.cv_folds,
        "tune_parameters": request.tune_parameters,
        "early_stopping_rounds": request.early_stopping_rounds,
        "objective": request.objective,
        "custom_param_grid": request.custom_param_grid,
    }

    try:
        project_id = create_project(project_data)

        # Add training to background tasks
        background_tasks.add_task(run_training_task, project_id, request)

        # Return immediately with project ID
        return {
            "project_id": project_id,
            "status": "training",
            "message": "Training started in background",
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to start training: {str(e)}"
        )


@router.get("/projects")
async def list_projects():
    """Get list of all projects"""
    projects = get_projects(limit=100)

    # Convert MongoDB documents to API response format
    formatted_projects = []
    for project in projects:
        formatted_project = {
            "id": project["_id"],
            "name": project["name"],
            "created_at": project["created_at"].isoformat()
            if isinstance(project["created_at"], datetime)
            else project["created_at"],
            "csv_filename": project["dataset"]["filename"],
            "target_column": project["dataset"]["target_column"],
            "feature_columns": project["dataset"]["feature_columns"],
            "model_params": project["config"],
            "status": project["status"],
        }

        # Add results if available
        if "results" in project:
            formatted_project.update(
                {
                    "auc_score": project["results"].get("test_auc"),
                    "accuracy": 0,  # Not used anymore
                    "feature_importance": project["results"].get("feature_importance"),
                    "confusion_matrix": [],  # Not used anymore
                    "lift_chart_data": project["results"].get("lift_chart_data"),
                }
            )
        else:
            formatted_project.update(
                {
                    "auc_score": None,
                    "accuracy": None,
                    "feature_importance": None,
                    "confusion_matrix": None,
                    "lift_chart_data": None,
                }
            )

        formatted_projects.append(formatted_project)

    return formatted_projects


@router.get("/projects/{project_id}")
async def get_project_details(project_id: str):
    """Get details of a specific project"""
    project = get_project(project_id)

    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Format response
    formatted_project = {
        "id": project["_id"],
        "name": project["name"],
        "created_at": project["created_at"].isoformat()
        if isinstance(project["created_at"], datetime)
        else project["created_at"],
        "csv_filename": project["dataset"]["filename"],
        "target_column": project["dataset"]["target_column"],
        "feature_columns": project["dataset"]["feature_columns"],
        "model_params": project["config"],
        "status": project["status"],
    }

    # Add results if available
    if "results" in project:
        formatted_project.update(
            {
                "auc_score": project["results"].get("test_auc"),
                "accuracy": 0,  # Not used anymore
                "feature_importance": project["results"].get("feature_importance"),
                "confusion_matrix": [],  # Not used anymore
                "lift_chart_data": project["results"].get("lift_chart_data"),
            }
        )
    else:
        formatted_project.update(
            {
                "auc_score": None,
                "accuracy": None,
                "feature_importance": None,
                "confusion_matrix": None,
                "lift_chart_data": None,
            }
        )

    return formatted_project


@router.get("/projects/{project_id}/logs")
async def get_project_training_logs(project_id: str):
    """Get all training logs for a project"""
    # Check if project exists
    project = get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Get all logs
    logs = get_training_logs(project_id, limit=1000)

    # Format logs
    formatted_logs = []
    for log in reversed(logs):  # Reverse to get chronological order
        formatted_logs.append(
            {
                "message": log["message"],
                "timestamp": log["timestamp"].isoformat()
                if isinstance(log["timestamp"], datetime)
                else log["timestamp"],
            }
        )

    return {"project_id": project_id, "logs": formatted_logs}


@router.get("/xgboost/param-info")
async def get_xgboost_param_info():
    """Get information about available XGBoost parameters for tuning"""
    return {
        "tunable_parameters": get_param_info(),
        "default_param_grid": DEFAULT_PARAM_GRID,
    }


@router.post("/xgboost/validate-param-grid")
async def validate_parameter_grid(param_grid: dict):
    """Validate a parameter grid for XGBoost hyperparameter tuning"""
    from xgb_params import validate_param_grid

    is_valid, errors = validate_param_grid(param_grid)

    if is_valid:
        return {"valid": True, "errors": []}
    else:
        return {"valid": False, "errors": errors}


@router.post("/check-feature-signal")
async def check_feature_signal(request: FeatureSignalRequest):
    """Calculate feature importance scores using univariate statistical tests"""
    try:
        # Load the dataset
        file_path = os.path.join(UPLOAD_DIR, request.csv_filename)
        df = pd.read_csv(file_path)

        # Get target
        y = df[request.target_column]

        # Prepare feature scores dictionary
        feature_scores = {}

        for feature in request.feature_columns:
            if feature == request.target_column:
                continue

            try:
                # Get feature data
                X = df[[feature]].copy()

                # Handle non-numeric features
                if X[feature].dtype == "object":
                    # First, handle missing values
                    imputer = SimpleImputer(strategy="constant", fill_value="missing")
                    X[feature] = imputer.fit_transform(X[[feature]]).ravel()

                    # Then encode categorical features using OrdinalEncoder
                    encoder = OrdinalEncoder(
                        handle_unknown="use_encoded_value", unknown_value=-1
                    )
                    X[feature] = encoder.fit_transform(X[[feature]]).ravel()
                else:
                    # For numeric features, impute missing values with median
                    if X[feature].isna().any():
                        imputer = SimpleImputer(strategy="median")
                        X[feature] = imputer.fit_transform(X[[feature]]).ravel()

                # Remove rows where target has NaN values
                mask = ~y.isna()
                X_clean = X[mask]
                y_clean = y[mask]

                if len(X_clean) < 10:  # Skip if too few samples
                    continue

                # Reshape for sklearn
                X_clean = X_clean.values.reshape(-1, 1)

                # Determine if target is binary or continuous
                is_binary = y_clean.nunique() <= 10

                if is_binary:
                    # Use f_classif for classification
                    f_scores, _ = f_classif(X_clean, y_clean)
                else:
                    # Use f_regression for regression
                    f_scores, _ = f_regression(X_clean, y_clean)

                # Handle NaN or infinite values
                f_score = f_scores[0]
                if np.isnan(f_score) or np.isinf(f_score):
                    f_score = 0.0

                feature_scores[feature] = float(f_score)

            except Exception as e:
                # Skip features that cause errors
                print(f"Error processing feature {feature}: {str(e)}")
                feature_scores[feature] = 0.0

        return feature_scores

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to check feature signal: {str(e)}"
        )


@router.delete("/projects/{project_id}")
async def delete_project(project_id: str):
    """Delete a project and its associated data"""
    try:
        # Import the delete function from mongo_utils
        from mongo_utils import delete_project as mongo_delete_project

        # Check if project exists
        project = get_project(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        # Delete the project from MongoDB
        success = mongo_delete_project(project_id)

        if success:
            # Optionally, delete associated model files
            model_file = f"models/{project_id}.json"
            if os.path.exists(model_file):
                os.remove(model_file)

            return {"message": "Project deleted successfully", "project_id": project_id}
        else:
            raise HTTPException(status_code=500, detail="Failed to delete project")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting project: {str(e)}")


@router.get("/projects/{project_id}/generate-code")
async def generate_project_code(project_id: str):
    """Generate standalone Python code to reproduce the XGBoost model training"""
    try:
        # Get project details
        project = get_project(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        # Check if training is complete
        if project["status"] != "completed":
            raise HTTPException(
                status_code=400, 
                detail="Code generation is only available for completed models"
            )
        
        # Extract necessary information
        dataset = project["dataset"]
        config = project["config"]
        results = project.get("results", {})
        
        # Get preprocessing artifacts
        preprocessing_artifacts = results.get("preprocessing_artifacts", {})
        
        # Get model parameters
        model_params = results.get("model_params", {})
        if "objective" in model_params:
            del model_params["objective"]  # Will be set separately
        if "eval_metric" in model_params:
            del model_params["eval_metric"]  # Will be set separately
        if "seed" in model_params:
            del model_params["seed"]  # Will be set separately
        if "nthread" in model_params:
            del model_params["nthread"]  # Will be set separately
        
        # Generate the code
        code = generate_training_code(
            csv_filename=dataset["filename"],
            feature_columns=dataset["feature_columns"],
            target_column=dataset["target_column"],
            test_size=config.get("test_size", 0.2),
            model_params=model_params,
            preprocessing_artifacts=preprocessing_artifacts,
            n_estimators=results.get("n_estimators", 100),
            objective=config.get("objective", "binary:logistic"),
            eval_metric=config.get("eval_metric", "auc")
        )
        
        return {
            "project_id": project_id,
            "project_name": project["name"],
            "code": code,
            "filename": f"{project['name'].replace(' ', '_').lower()}_training.py"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to generate code: {str(e)}"
        )
