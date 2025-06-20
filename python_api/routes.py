from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
import pandas as pd
import numpy as np
import json
import os
import asyncio
from datetime import datetime

try:
    from .xgb_modeling import train_xgboost_model
    from .models import TrainRequest
    from .config import UPLOAD_DIR
    from .mongo_utils import create_model, get_model, get_models, get_training_logs
except ImportError:
    from xgb_modeling import train_xgboost_model
    from models import TrainRequest
    from config import UPLOAD_DIR
    from mongo_utils import create_model, get_model, get_models, get_training_logs

# Create router instance
router = APIRouter(prefix="/api")


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


@router.post("/train")
async def train_model(request: TrainRequest):
    """Start model training with provided configuration"""
    # Create model document in MongoDB
    model_data = {
        "name": request.model_name,
        "csv_filename": request.csv_filename,
        "target_column": request.target_column,
        "feature_columns": request.feature_columns,
        "test_size": request.test_size,
        "cv_folds": request.cv_folds,
        "tune_parameters": request.tune_parameters,
        "early_stopping_rounds": request.early_stopping_rounds,
        "objective": request.objective,
    }

    try:
        model_id = create_model(model_data)

        # Start training
        result = train_xgboost_model(model_id, request)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@router.get("/models")
async def list_models():
    """Get list of all models"""
    models = get_models(limit=100)

    # Convert MongoDB documents to API response format
    formatted_models = []
    for model in models:
        formatted_model = {
            "id": model["_id"],
            "name": model["name"],
            "created_at": model["created_at"].isoformat()
            if isinstance(model["created_at"], datetime)
            else model["created_at"],
            "csv_filename": model["dataset"]["filename"],
            "target_column": model["dataset"]["target_column"],
            "feature_columns": model["dataset"]["feature_columns"],
            "model_params": model["config"],
            "status": model["status"],
        }

        # Add results if available
        if "results" in model:
            formatted_model.update(
                {
                    "auc_score": model["results"].get("test_auc"),
                    "accuracy": 0,  # Not used anymore
                    "feature_importance": model["results"].get("feature_importance"),
                    "confusion_matrix": [],  # Not used anymore
                    "lift_chart_data": model["results"].get("lift_chart_data"),
                }
            )
        else:
            formatted_model.update(
                {
                    "auc_score": None,
                    "accuracy": None,
                    "feature_importance": None,
                    "confusion_matrix": None,
                    "lift_chart_data": None,
                }
            )

        formatted_models.append(formatted_model)

    return formatted_models


@router.get("/models/{model_id}")
async def get_model_details(model_id: str):
    """Get details of a specific model"""
    model = get_model(model_id)

    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    # Format response
    formatted_model = {
        "id": model["_id"],
        "name": model["name"],
        "created_at": model["created_at"].isoformat()
        if isinstance(model["created_at"], datetime)
        else model["created_at"],
        "csv_filename": model["dataset"]["filename"],
        "target_column": model["dataset"]["target_column"],
        "feature_columns": model["dataset"]["feature_columns"],
        "model_params": model["config"],
        "status": model["status"],
    }

    # Add results if available
    if "results" in model:
        formatted_model.update(
            {
                "auc_score": model["results"].get("test_auc"),
                "accuracy": 0,  # Not used anymore
                "feature_importance": model["results"].get("feature_importance"),
                "confusion_matrix": [],  # Not used anymore
                "lift_chart_data": model["results"].get("lift_chart_data"),
            }
        )
    else:
        formatted_model.update(
            {
                "auc_score": None,
                "accuracy": None,
                "feature_importance": None,
                "confusion_matrix": None,
                "lift_chart_data": None,
            }
        )

    return formatted_model


@router.get("/models/{model_id}/progress")
async def stream_model_progress(model_id: str):
    """Stream training progress via Server-Sent Events"""

    async def generate():
        while True:
            # Get model status
            model = get_model(model_id)

            if not model:
                yield f"data: {json.dumps({'error': 'Model not found'})}\n\n"
                break

            status = model["status"]

            # Get latest logs
            logs = get_training_logs(model_id, limit=10)

            # Format logs for response
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

            yield f"data: {json.dumps({'status': status, 'logs': formatted_logs})}\n\n"

            if status in ["completed", "failed"]:
                break

            await asyncio.sleep(1)

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.get("/models/{model_id}/logs")
async def get_model_training_logs(model_id: str):
    """Get all training logs for a model"""
    # Check if model exists
    model = get_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    # Get all logs
    logs = get_training_logs(model_id, limit=1000)

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

    return {"model_id": model_id, "logs": formatted_logs}
