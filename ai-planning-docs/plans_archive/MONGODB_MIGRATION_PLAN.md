# MongoDB Atlas Migration Plan

## Current SQL Schema Limitations

The current SQLite implementation has several limitations:
- Fixed schema requires adding columns for new features
- Complex data structures (lists, nested objects) stored as JSON strings
- Growing table width as features are added
- Difficult to query nested data efficiently

## MongoDB Benefits

1. **Flexible Schema**: Documents can have different structures
2. **Native Support for Complex Types**: Arrays, nested objects, etc.
3. **Better for ML Metadata**: Easy to add new metrics without schema changes
4. **Scalability**: MongoDB Atlas provides automatic scaling

## Proposed MongoDB Collections

### 1. `models` Collection
```javascript
{
  _id: ObjectId,
  name: String,
  created_at: Date,
  updated_at: Date,
  status: String, // 'training', 'completed', 'failed'
  
  // Dataset info
  dataset: {
    filename: String,
    shape: [rows, cols],
    target_column: String,
    feature_columns: [String],
    upload_date: Date
  },
  
  // Training configuration
  config: {
    test_size: Number,
    cv_folds: Number,
    tune_parameters: Boolean,
    early_stopping_rounds: Number,
    objective: String,
    // Can easily add new config options here
  },
  
  // Results (only populated after training)
  results: {
    auc_score: Number,
    cv_auc_score: Number,
    feature_importance: {
      feature_name: importance_value,
      ...
    },
    lift_chart_data: [{
      bin: Number,
      avg_prediction: Number,
      actual_rate: Number,
      count: Number
    }],
    best_params: {
      max_depth: Number,
      learning_rate: Number,
      // etc...
    },
    n_estimators_used: Number,
    // Can add new metrics without schema change
  },
  
  // Model file reference
  model_file: String, // path to saved .json model
  
  // Extensible metadata
  metadata: {
    // Any additional data can go here
  }
}
```

### 2. `training_logs` Collection
```javascript
{
  _id: ObjectId,
  model_id: ObjectId,
  timestamp: Date,
  message: String,
  level: String, // 'info', 'warning', 'error'
  metadata: {} // Optional additional context
}
```

## Migration Code Example

### Connection Setup
```python
from pymongo import MongoClient
from datetime import datetime
import os

# MongoDB Atlas connection
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb+srv://...")
client = MongoClient(MONGODB_URI)
db = client.xgboost_models

# Collections
models_collection = db.models
logs_collection = db.training_logs
```

### Model Creation
```python
def create_model(model_data):
    model_doc = {
        "name": model_data["name"],
        "created_at": datetime.utcnow(),
        "status": "training",
        "dataset": {
            "filename": model_data["csv_filename"],
            "target_column": model_data["target_column"],
            "feature_columns": model_data["feature_columns"],
        },
        "config": {
            "test_size": model_data["test_size"],
            "cv_folds": model_data["cv_folds"],
            "tune_parameters": model_data["tune_parameters"],
            "early_stopping_rounds": model_data["early_stopping_rounds"],
            "objective": model_data.get("objective", "binary:logistic")
        }
    }
    
    result = models_collection.insert_one(model_doc)
    return str(result.inserted_id)
```

### Update Model Results
```python
def update_model_results(model_id, results):
    models_collection.update_one(
        {"_id": ObjectId(model_id)},
        {
            "$set": {
                "status": "completed",
                "updated_at": datetime.utcnow(),
                "results": results,
                "model_file": f"models/{model_id}.json"
            }
        }
    )
```

### Logging
```python
def log_message(model_id, message, level="info"):
    logs_collection.insert_one({
        "model_id": ObjectId(model_id),
        "timestamp": datetime.utcnow(),
        "message": message,
        "level": level
    })
```

## API Updates

### FastAPI with MongoDB
```python
from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorClient

# Async MongoDB client for FastAPI
motor_client = AsyncIOMotorClient(MONGODB_URI)
database = motor_client.xgboost_models

@router.get("/models/{model_id}")
async def get_model(model_id: str):
    model = await database.models.find_one({"_id": ObjectId(model_id)})
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    # Convert ObjectId to string for JSON serialization
    model["_id"] = str(model["_id"])
    return model
```

## Benefits of Migration

1. **Easier Feature Addition**: Add new metrics, visualizations, or metadata without schema changes
2. **Better Performance**: Native indexing on nested fields
3. **Aggregation Pipeline**: Powerful queries for analytics
4. **Time Series Data**: Better support for tracking model performance over time
5. **GridFS**: Store large model files directly in MongoDB

## Migration Steps

1. Set up MongoDB Atlas cluster
2. Update Python dependencies (`pymongo`, `motor` for async)
3. Create database helper functions
4. Update all route handlers
5. Migrate existing data (if needed)
6. Update frontend to handle MongoDB ObjectIds

## Future Extensibility

With MongoDB, adding new features becomes trivial:
- Model versioning
- Experiment tracking
- A/B testing results
- Real-time model performance monitoring
- Custom metadata per model type
- Nested hyperparameter configurations