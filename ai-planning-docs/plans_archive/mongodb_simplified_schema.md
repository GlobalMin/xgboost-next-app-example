# Simplified MongoDB Schema - Single Collection with Groupings

## Current Flat Structure Issues

- Mixing different concerns at the same level
- Hard to find related information
- Preprocessing artifacts mixed with results
- No clear separation between input configuration and output results

## Proposed Structure - Single "projects" Collection

```javascript
{
  "_id": ObjectId,
  "project_name": String,
  "created_at": Date,
  "updated_at": Date,
  "status": String,  // "training", "completed", "failed"
  
  // Group 1: Dataset Information
  "dataset": {
    "filename": String,
    "upload_date": Date,
    "row_count": Number,
    "column_count": Number,
    "target_column": String,
    "feature_columns": [String],
    "column_types": {
      "numeric": [String],
      "categorical": [String],
      "datetime": [String]
    },
    "missing_values": {
      "column_name": Number  // count of missing values per column
    }
  },
  
  // Group 2: Preprocessing Configuration
  "preprocessing": {
    "pipeline_definition": {
      "steps": [
        {
          "name": "datetime_converter",
          "type": "custom",
          "class": "DatetimeToString"
        },
        {
          "name": "column_transformer",
          "type": "sklearn.compose.ColumnTransformer",
          "transformers": [
            {
              "name": "numeric",
              "transformer": "Pipeline",
              "columns": ["PaymentTier", "JoiningYear"],
              "steps": [
                {"name": "imputer", "type": "SimpleImputer", "params": {"strategy": "constant", "fill_value": -9999}}
              ]
            },
            {
              "name": "categorical", 
              "transformer": "Pipeline",
              "columns": ["Gender", "City", "Education", "EverBenched"],
              "steps": [
                {"name": "imputer", "type": "SimpleImputer", "params": {"strategy": "constant", "fill_value": "missing"}},
                {"name": "encoder", "type": "OrdinalEncoder", "params": {"handle_unknown": "use_encoded_value", "unknown_value": -1}}
              ]
            }
          ]
        }
      ]
    },
    "feature_mappings": {
      "Gender": {"Female": 0, "Male": 1},
      "City": {"Bangalore": 0, "New Delhi": 1, "Pune": 2},
      "Education": {"Bachelors": 0, "Masters": 1, "PHD": 2},
      "EverBenched": {"No": 0, "Yes": 1}
    },
    "pipeline_code": String  // Generated Python code for reproducibility
  },
  
  // Group 3: Model Configuration
  "model_config": {
    "algorithm": "xgboost",  // Fixed for now
    "objective": String,     // "binary:logistic", "reg:squarederror", etc.
    "test_size": Number,
    "random_state": Number,
    "cv_folds": Number,
    "early_stopping_rounds": Number,
    "eval_metric": String
  },
  
  // Group 4: Hyperparameter Tuning
  "tuning": {
    "search_method": "grid_search",  // or "random_search"
    "param_grid": {
      "max_depth": [Number],
      "learning_rate": [Number],
      "n_estimators": [Number],
      "subsample": [Number],
      "colsample_bytree": [Number],
      "reg_lambda": [Number]
    },
    "cv_results": [
      {
        "params": Object,
        "mean_score": Number,
        "std_score": Number,
        "rank": Number
      }
    ],
    "best_params": {
      "max_depth": Number,
      "learning_rate": Number,
      "n_estimators": Number,
      "subsample": Number,
      "colsample_bytree": Number,
      "reg_lambda": Number
    },
    "best_score": Number,
    "tuning_duration_seconds": Number
  },
  
  // Group 5: Final Model
  "final_model": {
    "params": Object,  // Final parameters used
    "n_estimators": Number,  // Actual number after early stopping
    "training_duration_seconds": Number,
    "model_file_path": String,
    "model_size_bytes": Number,
    "feature_names": [String]  // Features in order expected by model
  },
  
  // Group 6: Evaluation Results
  "evaluation": {
    "metrics": {
      "train": {
        "auc": Number,
        "accuracy": Number,
        "precision": Number,
        "recall": Number,
        "f1": Number
      },
      "cv": {
        "auc_mean": Number,
        "auc_std": Number,
        "accuracy_mean": Number,
        "accuracy_std": Number
      },
      "test": {
        "auc": Number,
        "accuracy": Number,
        "precision": Number,
        "recall": Number,
        "f1": Number,
        "confusion_matrix": [[Number]]
      }
    },
    "feature_importance": {
      "feature_name": Number  // Importance score per feature
    },
    "lift_chart": {
      "bins": Number,
      "data": [
        {
          "bin": Number,
          "avg_prediction": Number,
          "actual_rate": Number,
          "lift": Number,
          "count": Number
        }
      ]
    }
  },
  
  // Group 7: Metadata and Settings
  "metadata": {
    "description": String,
    "tags": [String],
    "custom_fields": Object  // User-defined metadata
  },
  
  // Group 8: Error Handling (if failed)
  "error": {
    "message": String,
    "traceback": String,
    "failed_at": Date,
    "stage": String  // "preprocessing", "tuning", "training", "evaluation"
}cQA` QAQ`
}
```

## Benefits of This Structure

1. **Clear Organization**: Related data is grouped together
2. **Easy to Navigate**: Top-level keys make it clear what's where
3. **Minimal but Complete**: Keeps all necessary information without over-engineering
4. **Single Collection**: Simple to query and maintain
5. **Extensible**: Can add new groups or fields within groups easily

## Example Query Patterns

```javascript
// Find all completed projects
db.projects.find({"status": "completed"})

// Find projects with high test AUC
db.projects.find({"evaluation.metrics.test.auc": {$gte: 0.9}})

// Get preprocessing info for a project
db.projects.findOne({_id: projectId}, {preprocessing: 1})

// Find projects using specific features
db.projects.find({"dataset.feature_columns": "Gender"})

// Get just the final model parameters
db.projects.findOne({_id: projectId}, {"final_model.params": 1})
```

## Migration from Current Structure

```python
def migrate_project_structure(old_doc):
    """Transform old flat structure to new grouped structure"""
    return {
        "_id": old_doc["_id"],
        "project_name": old_doc["name"],
        "created_at": old_doc["created_at"],
        "updated_at": old_doc["updated_at"],
        "status": old_doc["status"],
        
        "dataset": {
            "filename": old_doc["dataset"]["filename"],
            "upload_date": old_doc["dataset"]["upload_date"],
            "target_column": old_doc["dataset"]["target_column"],
            "feature_columns": old_doc["dataset"]["feature_columns"],
            "column_types": {
                "numeric": old_doc["results"]["preprocessing_artifacts"]["numeric_columns"],
                "categorical": old_doc["results"]["preprocessing_artifacts"]["categorical_columns"],
                "datetime": old_doc["results"]["preprocessing_artifacts"]["datetime_columns"]
            }
        },
        
        "preprocessing": {
            "pipeline_definition": extract_pipeline_definition(old_doc),
            "feature_mappings": extract_feature_mappings(old_doc),
            "pipeline_code": old_doc["results"]["preprocessing_artifacts"]["pipeline_code"]
        },
        
        "model_config": {
            "algorithm": "xgboost",
            "objective": old_doc["config"]["objective"],
            "test_size": old_doc["config"]["test_size"],
            "cv_folds": old_doc["config"]["cv_folds"],
            "early_stopping_rounds": old_doc["config"]["early_stopping_rounds"]
        },
        
        "tuning": {
            "best_params": old_doc["results"]["model_params"],
            "best_score": old_doc["results"]["cv_auc"]
        },
        
        "final_model": {
            "params": old_doc["results"]["model_params"],
            "n_estimators": old_doc["results"]["n_estimators"],
            "model_file_path": old_doc["model_file"]
        },
        
        "evaluation": {
            "metrics": {
                "cv": {"auc_mean": old_doc["results"]["cv_auc"]},
                "test": {"auc": old_doc["results"]["test_auc"]}
            },
            "feature_importance": old_doc["results"]["feature_importance"],
            "lift_chart": {
                "bins": 10,
                "data": old_doc["results"]["lift_chart_data"]
            }
        },
        
        "metadata": old_doc.get("metadata", {}),
        "error": {"message": old_doc["error"]} if "error" in old_doc else None
    }
```
