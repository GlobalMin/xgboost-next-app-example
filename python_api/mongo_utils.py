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
DATABASE_NAME = os.getenv("MONGODB_DATABASE", "xgboost_models")

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
                self._client.admin.command('ping')
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
MODELS_COLLECTION = os.getenv("MODELS_COLLECTION", "xgboost_models")
LOGS_COLLECTION = os.getenv("LOGS_COLLECTION", "training_logs")

# Helper functions
def create_model(model_data: Dict[str, Any]) -> str:
    """Create a new model document in MongoDB"""
    db = mongo_conn.get_database()
    
    model_doc = {
        "name": model_data["name"],
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
        "status": "training",
        "dataset": {
            "filename": model_data["csv_filename"],
            "target_column": model_data["target_column"],
            "feature_columns": model_data["feature_columns"],
            "upload_date": datetime.utcnow()
        },
        "config": {
            "test_size": model_data.get("test_size", 0.2),
            "cv_folds": model_data.get("cv_folds", 5),
            "tune_parameters": model_data.get("tune_parameters", False),
            "early_stopping_rounds": model_data.get("early_stopping_rounds", 50),
            "objective": model_data.get("objective", "binary:logistic")
        },
        "metadata": {}
    }
    
    result = db[MODELS_COLLECTION].insert_one(model_doc)
    return str(result.inserted_id)

def update_model_status(model_id: str, status: str) -> bool:
    """Update model status"""
    db = mongo_conn.get_database()
    
    result = db[MODELS_COLLECTION].update_one(
        {"_id": ObjectId(model_id)},
        {
            "$set": {
                "status": status,
                "updated_at": datetime.utcnow()
            }
        }
    )
    return result.modified_count > 0

def update_model_results(model_id: str, results: Dict[str, Any]) -> bool:
    """Update model with training results"""
    db = mongo_conn.get_database()
    
    # Ensure all numeric values are JSON serializable
    clean_results = json.loads(json.dumps(results, default=str))
    
    result = db[MODELS_COLLECTION].update_one(
        {"_id": ObjectId(model_id)},
        {
            "$set": {
                "status": "completed",
                "updated_at": datetime.utcnow(),
                "results": clean_results,
                "model_file": f"models/{model_id}.json"
            }
        }
    )
    return result.modified_count > 0

def get_model(model_id: str) -> Optional[Dict[str, Any]]:
    """Get a model by ID"""
    db = mongo_conn.get_database()
    
    model = db[MODELS_COLLECTION].find_one({"_id": ObjectId(model_id)})
    if model:
        model["_id"] = str(model["_id"])
    return model

def get_models(limit: int = 100, skip: int = 0) -> List[Dict[str, Any]]:
    """Get models with pagination"""
    db = mongo_conn.get_database()
    
    models = []
    cursor = db[MODELS_COLLECTION].find().sort("created_at", -1).skip(skip).limit(limit)
    
    for model in cursor:
        model["_id"] = str(model["_id"])
        models.append(model)
    
    return models

def log_training_message(model_id: str, message: str, level: str = "info") -> str:
    """Log a training message"""
    db = mongo_conn.get_database()
    
    log_doc = {
        "model_id": model_id,
        "timestamp": datetime.utcnow(),
        "message": message,
        "level": level
    }
    
    result = db[LOGS_COLLECTION].insert_one(log_doc)
    return str(result.inserted_id)

def get_training_logs(model_id: str, limit: int = 100) -> List[Dict[str, Any]]:
    """Get training logs for a model"""
    db = mongo_conn.get_database()
    
    logs = []
    cursor = db[LOGS_COLLECTION].find(
        {"model_id": model_id}
    ).sort("timestamp", -1).limit(limit)
    
    for log in cursor:
        log["_id"] = str(log["_id"])
        logs.append(log)
    
    return logs

def add_model_metadata(model_id: str, key: str, value: Any) -> bool:
    """Add metadata to a model"""
    db = mongo_conn.get_database()
    
    result = db[MODELS_COLLECTION].update_one(
        {"_id": ObjectId(model_id)},
        {
            "$set": {
                f"metadata.{key}": value,
                "updated_at": datetime.utcnow()
            }
        }
    )
    return result.modified_count > 0

def cleanup_failed_model(model_id: str, error_message: str) -> bool:
    """Mark a model as failed and log the error"""
    db = mongo_conn.get_database()
    
    # Update model status
    db[MODELS_COLLECTION].update_one(
        {"_id": ObjectId(model_id)},
        {
            "$set": {
                "status": "failed",
                "updated_at": datetime.utcnow(),
                "error": error_message
            }
        }
    )
    
    # Log the error
    log_training_message(model_id, f"Training failed: {error_message}", level="error")
    
    return True