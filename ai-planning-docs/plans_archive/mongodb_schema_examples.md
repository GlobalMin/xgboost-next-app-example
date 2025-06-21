# MongoDB Schema Examples - Before and After

## Current Structure (Before)

```json
{
  "_id": {"$oid": "6856f57bdb57ce1f6256ef8b"},
  "name": "Employee_LeaveOrNot_20250621_181003",
  "created_at": {"$date": "2025-06-21T18:10:03.496Z"},
  "updated_at": {"$date": "2025-06-21T18:10:33.698Z"},
  "status": "completed",
  "dataset": {
    "filename": "Employee.csv",
    "target_column": "LeaveOrNot",
    "feature_columns": ["Gender", "City", "PaymentTier", "JoiningYear", "Education", "EverBenched"],
    "upload_date": {"$date": "2025-06-21T18:10:03.496Z"}
  },
  "config": {
    "test_size": 0.2,
    "cv_folds": 3,
    "early_stopping_rounds": 50,
    "objective": "binary:logistic"
  },
  "metadata": {},
  "model_file": "models/6856f57bdb57ce1f6256ef8b.json",
  "results": {
    "test_auc": 0.8665,
    "cv_auc": 0.8638,
    "feature_importance": {
      "Gender": 9.48,
      "City": 10.20,
      "PaymentTier": 11.03,
      "JoiningYear": 29.14,
      "Education": 11.26,
      "EverBenched": 2.11
    },
    "lift_chart_data": [...],
    "model_params": {
      "max_depth": 4,
      "learning_rate": 0.1,
      "lambda": 0.5,
      "subsample": 0.8,
      "colsample_bytree": 0.8
    },
    "n_estimators": 62,
    "preprocessing_artifacts": {
      "numeric_columns": ["PaymentTier", "JoiningYear"],
      "categorical_columns": ["Gender", "City", "Education", "EverBenched"],
      "encoders": {...},
      "imputers": {...},
      "pipeline_code": "..."
    }
  }
}
```

## Improved Structure (After)

### 1. Projects Collection

```json
{
  "_id": {"$oid": "6856f57bdb57ce1f6256ef8b"},
  "name": "Employee Attrition Analysis",
  "description": "Predict employee turnover using HR data",
  "type": "classification",
  "created_at": {"$date": "2025-06-21T18:10:03.496Z"},
  "updated_at": {"$date": "2025-06-21T18:10:33.698Z"},
  "created_by": "user_123",
  "tags": ["hr-analytics", "employee-retention", "binary-classification"],
  "status": {
    "state": "completed",
    "message": "Model training completed successfully",
    "updated_at": {"$date": "2025-06-21T18:10:33.698Z"}
  },
  "settings": {
    "auto_versioning": true,
    "notifications_enabled": true,
    "retention_days": 90
  }
}
```

### 2. Experiments Collection

```json
{
  "_id": {"$oid": "6856f580db57ce1f6256ef8c"},
  "project_id": {"$oid": "6856f57bdb57ce1f6256ef8b"},
  "experiment_name": "baseline_xgboost",
  "version": 1,
  "created_at": {"$date": "2025-06-21T18:10:03.496Z"},
  "created_by": "user_123",
  "status": "completed",
  "description": "Initial baseline model with default parameters",
  "config": {
    "dataset": {
      "dataset_id": {"$oid": "6856f570db57ce1f6256ef8a"},
      "target_column": "LeaveOrNot",
      "feature_columns": ["Gender", "City", "PaymentTier", "JoiningYear", "Education", "EverBenched"],
      "preprocessing_id": {"$oid": "6856f590db57ce1f6256ef8d"},
      "sampling": {
        "method": "none",
        "ratio": 1.0
      }
    },
    "model": {
      "algorithm": "xgboost",
      "framework_version": "1.7.5",
      "objective": "binary:logistic",
      "hyperparameters": {
        "max_depth": 4,
        "learning_rate": 0.1,
        "reg_lambda": 0.5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "n_estimators": 62
      },
      "hyperparameter_search": {
        "method": "grid_search",
        "param_grid": {
          "max_depth": [3, 4, 5],
          "learning_rate": [0.01, 0.1, 0.3]
        },
        "scoring": "roc_auc"
      }
    },
    "training": {
      "test_size": 0.2,
      "random_state": 42,
      "stratify": true,
      "early_stopping_rounds": 50,
      "eval_metric": "auc",
      "cv_strategy": {
        "method": "stratified_kfold",
        "folds": 3,
        "shuffle": true
      }
    }
  },
  "results": {
    "metrics": {
      "train": {
        "auc": 0.9123,
        "accuracy": 0.8756,
        "precision": 0.8234,
        "recall": 0.8912,
        "f1": 0.8560
      },
      "validation": {
        "auc": 0.8638,
        "accuracy": 0.8234,
        "precision": 0.7812,
        "recall": 0.8456,
        "f1": 0.8121,
        "std_auc": 0.0234  
      },
      "test": {
        "auc": 0.8665,
        "accuracy": 0.8312,
        "precision": 0.7923,
        "recall": 0.8534,
        "f1": 0.8217,
        "confusion_matrix": [[234, 56], [45, 189]]
      }
    },
    "feature_importance": {
      "JoiningYear": 29.14,
      "Education": 11.26,
      "PaymentTier": 11.03,
      "City": 10.20,
      "Gender": 9.48,
      "EverBenched": 2.11
    },
    "training_history": [
      {"iteration": 0, "train_auc": 0.6234, "valid_auc": 0.6123},
      {"iteration": 10, "train_auc": 0.7845, "valid_auc": 0.7623},
      {"iteration": 20, "train_auc": 0.8456, "valid_auc": 0.8234}
    ],
    "lift_analysis": {
      "bins": 10,
      "data": [
        {"bin": 1, "avg_prediction": 0.0678, "actual_rate": 0.0385, "lift": 0.568, "count": 104},
        {"bin": 10, "avg_prediction": 0.9744, "actual_rate": 1.0, "lift": 1.027, "count": 80}
      ]
    },
    "performance": {
      "training_time_seconds": 12.34,
      "prediction_time_ms": 0.23,
      "model_size_bytes": 45678,
      "peak_memory_mb": 234.5
    }
  },
  "artifacts": {
    "model_path": "s3://models/6856f580db57ce1f6256ef8c/model.json",
    "preprocessing_path": "s3://models/6856f580db57ce1f6256ef8c/preprocessor.pkl",
    "reports_path": "s3://models/6856f580db57ce1f6256ef8c/reports/",
    "checksum": "sha256:abc123..."
  },
  "tags": ["baseline", "grid-search", "production-candidate"]
}
```

### 3. Datasets Collection

```json
{
  "_id": {"$oid": "6856f570db57ce1f6256ef8a"},
  "name": "Employee HR Dataset",
  "filename": "Employee.csv",
  "file_path": "s3://datasets/hr/Employee.csv",
  "size_bytes": 234567,
  "row_count": 4653,
  "column_count": 10,
  "columns": [
    {
      "name": "Gender",
      "type": "categorical",
      "dtype": "object",
      "missing_count": 0,
      "unique_count": 2,
      "categories": ["Male", "Female"],
      "distribution": {"Male": 0.65, "Female": 0.35}
    },
    {
      "name": "JoiningYear",
      "type": "numeric",
      "dtype": "int64",
      "missing_count": 23,
      "unique_count": 8,
      "stats": {
        "min": 2012,
        "max": 2020,
        "mean": 2016.5,
        "std": 2.3,
        "quartiles": [2014, 2016, 2018]
      }
    },
    {
      "name": "LeaveOrNot",
      "type": "categorical",
      "dtype": "int64",
      "missing_count": 0,
      "unique_count": 2,
      "categories": [0, 1],
      "distribution": {0: 0.72, 1: 0.28},
      "target_candidate": true
    }
  ],
  "checksum": "sha256:1234567890abcdef",
  "uploaded_at": {"$date": "2025-06-21T18:00:00.000Z"},
  "uploaded_by": "user_123",
  "validation": {
    "status": "passed",
    "issues": [],
    "warnings": ["23 missing values in JoiningYear column"]
  },
  "metadata": {
    "source": "HR Department",
    "collection_date": "2025-06-01",
    "privacy_level": "internal"
  }
}
```

### 4. Preprocessing Pipelines Collection

```json
{
  "_id": {"$oid": "6856f590db57ce1f6256ef8d"},
  "name": "Standard HR Preprocessing v1",
  "version": "1.0.0",
  "created_at": {"$date": "2025-06-21T18:10:00.000Z"},
  "created_by": "user_123",
  "dataset_id": {"$oid": "6856f570db57ce1f6256ef8a"},
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
  "output_schema": {
    "features": ["PaymentTier", "JoiningYear", "Gender", "City", "Education", "EverBenched"],
    "dtypes": ["float64", "float64", "float64", "float64", "float64", "float64"]
  },
  "code_hash": "sha256:abcd1234...",
  "dependencies": {
    "scikit-learn": "1.3.0",
    "pandas": "2.0.3",
    "numpy": "1.24.3"
  }
}
```

### 5. Model Registry Collection (Bonus)

```json
{
  "_id": {"$oid": "6856f5a0db57ce1f6256ef8e"},
  "experiment_id": {"$oid": "6856f580db57ce1f6256ef8c"},
  "project_id": {"$oid": "6856f57bdb57ce1f6256ef8b"},
  "model_name": "employee_attrition_xgboost",
  "version": "1.0.0",
  "stage": "staging",  
  "registered_at": {"$date": "2025-06-21T18:15:00.000Z"},
  "registered_by": "user_123",
  "description": "Initial production model for employee attrition",
  "metrics_summary": {
    "test_auc": 0.8665,
    "test_accuracy": 0.8312
  },
  "deployment": {
    "endpoint": "https://api.company.com/models/employee-attrition/v1",
    "last_deployed": {"$date": "2025-06-21T19:00:00.000Z"},
    "deployment_status": "active"
  },
  "monitoring": {
    "drift_detection": true,
    "performance_tracking": true,
    "alert_thresholds": {
      "auc_drop": 0.05,
      "prediction_latency_ms": 100
    }
  },
  "lineage": {
    "dataset_version": "6856f570db57ce1f6256ef8a",
    "preprocessing_version": "6856f590db57ce1f6256ef8d",
    "parent_model": null
  }
}

```