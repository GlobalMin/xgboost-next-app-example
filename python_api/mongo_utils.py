"""MongoDB utility functions for database operations"""

import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from bson import ObjectId
import json

# MongoDB connection
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
DATABASE_NAME = os.getenv("MONGODB_DATABASE", "xgboost_next_app")


class MongoConnection:
    """Singleton MongoDB connection manager"""

    _instance = None
    _client = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MongoConnection, cls).__new__(cls)
        return cls._instance

    def get_client(self):
        if self._client is None:
            try:
                self._client = MongoClient(MONGODB_URI)
                # Test connection
                self._client.admin.command("ping")
                print(f"Connected to MongoDB at {MONGODB_URI}")
            except ConnectionFailure as e:
                print(f"Failed to connect to MongoDB: {e}")
                raise
        return self._client

    def get_database(self):
        client = self.get_client()
        return client[DATABASE_NAME]


# Global connection instance
mongo_conn = MongoConnection()

# Collection names
PROJECTS_COLLECTION = os.getenv("PROJECTS_COLLECTION", "projects")
LOGS_COLLECTION = os.getenv("LOGS_COLLECTION", "training_logs")


# Helper functions
def create_project(project_data: Dict[str, Any]) -> str:
    """Create a new project document in MongoDB with new schema"""
    db = mongo_conn.get_database()

    project_doc = {
        "project_name": project_data["name"],
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
        "status": "training",
        # Group 1: Dataset Information
        "dataset": {
            "filename": project_data["csv_filename"],
            "upload_date": datetime.utcnow(),
            "target_column": project_data["target_column"],
            "feature_columns": project_data["feature_columns"],
            # These will be populated during preprocessing
            "column_types": {"numeric": [], "categorical": [], "datetime": []},
            "missing_values": {},
            "row_count": None,
            "column_count": None,
        },
        # Group 2: Model Configuration
        "model_config": {
            "algorithm": "xgboost",
            "objective": project_data.get("objective", "binary:logistic"),
            "test_size": project_data.get("test_size", 0.2),
            "random_state": project_data.get("random_state", 42),
            "cv_folds": project_data.get("cv_folds", 5),
            "early_stopping_rounds": project_data.get("early_stopping_rounds", 50),
            "eval_metric": "auc",  # Default for binary classification
        },
        # Group 3: Metadata
        "metadata": project_data.get("metadata", {}),
        # Groups that will be populated later:
        # - preprocessing
        # - tuning
        # - final_model
        # - evaluation
        # - error (if failed)
    }

    # Add custom hyperparameter grid if provided
    if "custom_param_grid" in project_data:
        project_doc["tuning"] = {
            "search_method": "grid_search",
            "param_grid": project_data["custom_param_grid"],
        }

    result = db[PROJECTS_COLLECTION].insert_one(project_doc)
    return str(result.inserted_id)


def update_project_status(project_id: str, status: str) -> bool:
    """Update project status"""
    db = mongo_conn.get_database()

    result = db[PROJECTS_COLLECTION].update_one(
        {"_id": ObjectId(project_id)},
        {"$set": {"status": status, "updated_at": datetime.utcnow()}},
    )
    return result.modified_count > 0


def update_project_results(project_id: str, results: Dict[str, Any]) -> bool:
    """Update project with training results using new schema"""
    db = mongo_conn.get_database()

    # Ensure all numeric values are JSON serializable
    clean_results = json.loads(json.dumps(results, default=str))

    # Reorganize results into new schema structure
    update_doc = {
        "status": "completed",
        "updated_at": datetime.utcnow(),
        # Update dataset info with discovered column types
        "dataset.column_types": clean_results.get("preprocessing_artifacts", {}).get(
            "column_types",
            {
                "numeric": clean_results.get("preprocessing_artifacts", {}).get(
                    "numeric_columns", []
                ),
                "categorical": clean_results.get("preprocessing_artifacts", {}).get(
                    "categorical_columns", []
                ),
                "datetime": clean_results.get("preprocessing_artifacts", {}).get(
                    "datetime_columns", []
                ),
            },
        ),
        # Preprocessing group
        "preprocessing": {
            "pipeline_definition": clean_results.get("preprocessing_artifacts", {}).get(
                "pipeline_definition", {}
            ),
            "feature_mappings": clean_results.get("preprocessing_artifacts", {}).get(
                "encoders", {}
            ),
            "pipeline_code": clean_results.get("preprocessing_artifacts", {}).get(
                "pipeline_code", ""
            ),
        },
        # Tuning group (if hyperparameter tuning was done)
        "tuning.best_params": clean_results.get("model_params", {}),
        "tuning.best_score": clean_results.get("cv_auc", None),
        # Final model group
        "final_model": {
            "params": clean_results.get("model_params", {}),
            "n_estimators": clean_results.get("n_estimators", None),
            "model_file_path": f"models/{project_id}.json",
            "feature_names": clean_results.get("feature_names", []),
        },
        # Evaluation group
        "evaluation": {
            "metrics": {
                "cv": {
                    "auc_mean": clean_results.get("cv_auc", None),
                    "auc_std": clean_results.get("cv_auc_std", None),
                },
                "test": {
                    "auc": clean_results.get("test_auc", None),
                    "accuracy": clean_results.get("test_accuracy", None),
                    "precision": clean_results.get("test_precision", None),
                    "recall": clean_results.get("test_recall", None),
                    "f1": clean_results.get("test_f1", None),
                    "confusion_matrix": clean_results.get("confusion_matrix", None),
                },
            },
            "feature_importance": clean_results.get("feature_importance", {}),
            "lift_chart": {
                "bins": 10,
                "data": clean_results.get("lift_chart_data", []),
                "data_multi": clean_results.get("lift_chart_data_multi", {}),
            },
            "train_size": clean_results.get("train_size"),
            "test_size": clean_results.get("test_size"),
            "dataset_stats": clean_results.get("dataset_info", {}),
        },
    }

    result = db[PROJECTS_COLLECTION].update_one(
        {"_id": ObjectId(project_id)}, {"$set": update_doc}
    )
    return result.modified_count > 0


def get_project(project_id: str) -> Optional[Dict[str, Any]]:
    """Get a project by ID"""
    db = mongo_conn.get_database()

    project = db[PROJECTS_COLLECTION].find_one({"_id": ObjectId(project_id)})
    if project:
        project["_id"] = str(project["_id"])
    return project


def get_projects(limit: int = 100, skip: int = 0) -> List[Dict[str, Any]]:
    """Get projects with pagination"""
    db = mongo_conn.get_database()

    projects = []
    cursor = (
        db[PROJECTS_COLLECTION].find().sort("created_at", -1).skip(skip).limit(limit)
    )

    for project in cursor:
        project["_id"] = str(project["_id"])
        projects.append(project)

    return projects


def log_training_message(project_id: str, message: str, level: str = "info") -> str:
    """Log a training message

    DEPRECATED: This function is kept for backward compatibility.
    Use the logging module with model_id in extra fields instead.
    """
    import warnings
    from logging_config import get_logger

    warnings.warn(
        "log_training_message is deprecated. Use logging module instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    # Use the new logging system
    logger = get_logger("xgboost_training")
    log_method = getattr(logger, level.lower(), logger.info)

    # Log using the new system with project_id
    log_method(message, extra={"project_id": project_id})

    # Return a dummy ID for compatibility
    return "logged_via_new_system"


def get_training_logs(project_id: str, limit: int = 100) -> List[Dict[str, Any]]:
    """Get training logs for a project from JSON files"""
    from logging_config import read_project_logs

    # Read logs from JSON file instead of MongoDB
    logs = read_project_logs(project_id, limit=limit)

    # Add a dummy _id field for compatibility with existing code
    for i, log in enumerate(logs):
        if "_id" not in log:
            log["_id"] = f"log_{i}"

    return logs


def add_project_metadata(project_id: str, key: str, value: Any) -> bool:
    """Add metadata to a project"""
    db = mongo_conn.get_database()

    result = db[PROJECTS_COLLECTION].update_one(
        {"_id": ObjectId(project_id)},
        {"$set": {f"metadata.{key}": value, "updated_at": datetime.utcnow()}},
    )
    return result.modified_count > 0


def cleanup_failed_project(
    project_id: str, error_message: str, stage: str = "unknown"
) -> bool:
    """Mark a project as failed and log the error"""
    from logging_config import get_logger
    import traceback

    logger = get_logger(__name__)
    db = mongo_conn.get_database()

    # Update project status with new error structure
    db[PROJECTS_COLLECTION].update_one(
        {"_id": ObjectId(project_id)},
        {
            "$set": {
                "status": "failed",
                "updated_at": datetime.utcnow(),
                "error": {
                    "message": error_message,
                    "traceback": traceback.format_exc(),
                    "failed_at": datetime.utcnow(),
                    "stage": stage,
                },
            }
        },
    )

    # Log the error using the new logging system
    logger.error(
        f"Training failed at {stage}: {error_message}", extra={"project_id": project_id}
    )

    return True


def update_project_dataset_info(project_id: str, dataset_info: Dict[str, Any]) -> bool:
    """Update dataset information after initial analysis"""
    db = mongo_conn.get_database()

    update_fields = {}
    for key, value in dataset_info.items():
        update_fields[f"dataset.{key}"] = value
    update_fields["updated_at"] = datetime.utcnow()

    result = db[PROJECTS_COLLECTION].update_one(
        {"_id": ObjectId(project_id)}, {"$set": update_fields}
    )
    return result.modified_count > 0


def update_project_preprocessing(
    project_id: str, preprocessing_info: Dict[str, Any]
) -> bool:
    """Update preprocessing information"""
    db = mongo_conn.get_database()

    result = db[PROJECTS_COLLECTION].update_one(
        {"_id": ObjectId(project_id)},
        {
            "$set": {
                "preprocessing": preprocessing_info,
                "updated_at": datetime.utcnow(),
            }
        },
    )
    return result.modified_count > 0


def update_project_tuning_results(
    project_id: str, tuning_results: Dict[str, Any]
) -> bool:
    """Update hyperparameter tuning results"""
    db = mongo_conn.get_database()

    result = db[PROJECTS_COLLECTION].update_one(
        {"_id": ObjectId(project_id)},
        {"$set": {"tuning": tuning_results, "updated_at": datetime.utcnow()}},
    )
    return result.modified_count > 0


def delete_project(project_id: str) -> bool:
    """Delete a project and all associated data"""
    try:
        db = mongo_conn.get_database()

        # Delete the project document
        result = db[PROJECTS_COLLECTION].delete_one({"_id": ObjectId(project_id)})

        # Delete associated training logs
        db[LOGS_COLLECTION].delete_many({"project_id": project_id})

        return result.deleted_count > 0
    except Exception as e:
        print(f"Error deleting project {project_id}: {e}")
        return False
