# MongoDB Schema Improvements

## Current Issues

1. **Inconsistent Field Naming**: Mix of snake_case and flat structure
2. **Large Embedded Documents**: `preprocessing_artifacts` can be very large
3. **No Versioning**: Cannot track model iterations or experiments
4. **Limited Metadata**: Fixed structure doesn't allow for flexible metadata
5. **Missing Audit Trail**: Only basic timestamps, no change history
6. **No Schema Validation**: No constraints at database level
7. **Redundant Data**: Model file path could be derived

## Proposed Improvements

### 1. Separate Collections Structure

```javascript
// projects collection - core project info
{
  "_id": ObjectId,
  "name": String,
  "description": String,  // NEW: project description
  "type": String,  // NEW: "classification" or "regression"
  "created_at": Date,
  "updated_at": Date,
  "created_by": String,  // NEW: user ID
  "tags": [String],  // NEW: project tags
  "status": {
    "state": String,  // "draft", "training", "completed", "failed"
    "message": String,  // status message
    "updated_at": Date
  }
}

// experiments collection - individual model runs
{
  "_id": ObjectId,
  "project_id": ObjectId,  // reference to projects
  "experiment_name": String,
  "version": Number,  // NEW: auto-incrementing version
  "created_at": Date,
  "status": String,
  "config": {
    "dataset": {
      "file_id": ObjectId,  // reference to datasets collection
      "target_column": String,
      "feature_columns": [String],
      "preprocessing_version": String  // reference to preprocessing
    },
    "model": {
      "algorithm": String,  // "xgboost", future: "lightgbm", etc.
      "objective": String,
      "hyperparameters": Object,  // flexible schema
      "cv_strategy": {
        "method": String,  // "kfold", "stratified", etc.
        "folds": Number,
        "shuffle": Boolean
      }
    },
    "training": {
      "test_size": Number,
      "random_state": Number,
      "early_stopping_rounds": Number,
      "eval_metric": String
    }
  },
  "results": {
    "metrics": {
      "train": Object,  // flexible metrics
      "validation": Object,
      "test": Object
    },
    "feature_importance": Object,
    "training_history": [Object],  // epoch-by-epoch metrics
    "duration_seconds": Number,
    "model_size_bytes": Number
  },
  "artifacts": {
    "model_path": String,
    "preprocessing_path": String,
    "reports_path": String
  }
}

// datasets collection - reusable datasets
{
  "_id": ObjectId,
  "name": String,
  "filename": String,
  "file_path": String,
  "size_bytes": Number,
  "row_count": Number,
  "column_count": Number,
  "columns": [{
    "name": String,
    "type": String,
    "missing_count": Number,
    "unique_count": Number,
    "stats": Object  // min, max, mean, etc. for numeric
  }],
  "checksum": String,  // file integrity
  "uploaded_at": Date,
  "uploaded_by": String
}

// preprocessing_pipelines collection
{
  "_id": ObjectId,
  "name": String,
  "version": String,
  "created_at": Date,
  "pipeline_definition": Object,  // full pipeline config
  "feature_definitions": Object,
  "encoders": Object,
  "imputers": Object,
  "code_hash": String  // for versioning
}
```

### 2. Schema Validation

Add MongoDB schema validation for each collection:

```javascript
db.createCollection("projects", {
  validator: {
    $jsonSchema: {
      bsonType: "object",
      required: ["name", "type", "created_at", "status"],
      properties: {
        name: {
          bsonType: "string",
          minLength: 1,
          maxLength: 100
        },
        type: {
          enum: ["classification", "regression"]
        },
        status: {
          bsonType: "object",
          required: ["state"],
          properties: {
            state: {
              enum: ["draft", "training", "completed", "failed"]
            }
          }
        }
      }
    }
  }
});
```

### 3. Indexes for Performance

```javascript
// Projects
db.projects.createIndex({ "created_at": -1 });
db.projects.createIndex({ "status.state": 1 });
db.projects.createIndex({ "tags": 1 });

// Experiments  
db.experiments.createIndex({ "project_id": 1, "version": -1 });
db.experiments.createIndex({ "created_at": -1 });
db.experiments.createIndex({ "results.metrics.test.auc": -1 });  // for sorting by performance

// Datasets
db.datasets.createIndex({ "checksum": 1 });  // detect duplicates
db.datasets.createIndex({ "uploaded_at": -1 });
```

### 4. Benefits of New Structure

1. **Separation of Concerns**: Projects, experiments, datasets are separate
2. **Reusability**: Datasets and preprocessing can be reused
3. **Versioning**: Track multiple experiments per project
4. **Scalability**: Large artifacts stored separately
5. **Flexibility**: Schema allows for different ML algorithms
6. **Audit Trail**: Can track all experiments and changes
7. **Performance**: Better indexing and smaller documents
8. **Data Integrity**: Schema validation ensures consistency

### 5. Migration Strategy

1. Create new collections with validation
2. Write migration script to transform existing data
3. Update API to support both schemas during transition
4. Gradually migrate clients to new schema
5. Archive old collection after migration

### 6. API Changes

Update API responses to maintain backward compatibility:

```python
def transform_to_legacy_format(project, experiment):
    """Transform new schema to old API format"""
    return {
        "id": str(project["_id"]),
        "name": project["name"],
        "status": project["status"]["state"],
        "dataset": {
            "filename": experiment["config"]["dataset"]["filename"],
            # ... map other fields
        },
        "results": experiment.get("results", {}),
        # ... rest of mapping
    }

```