"use client";

import React, { useState, useEffect } from "react";
import {
  CheckCircle,
  AlertCircle,
  TrendingUp,
  FileText,
  ChevronDown,
  ChevronUp,
} from "lucide-react";
import { LiftChart } from "./lift-chart";

interface ModelResultsProps {
  model: {
    id: number;
    name: string;
    created_at: string;
    auc_score: number;
    cv_auc_score?: number;
    cv_auc_std?: number;
    test_auc_score?: number;
    feature_importance: Record<string, number>;
    best_params?: Record<string, any>;
    n_estimators_used?: number;
    lift_chart_data?: any[];
    lift_chart_data_multi?: {
      all?: any[];
      cv?: any[];
      test?: any[];
    };
    status: string;
    train_size?: number;
    test_size?: number;
    dataset_stats?: {
      total_rows?: number;
      target_rate?: number;
      total_columns?: number;
      n_features_used?: number;
    };
    csv_filename?: string;
    target_column?: string;
    feature_columns?: string[];
  };
}

export function ModelResults({ model }: ModelResultsProps) {
  const [logs, setLogs] = useState<{ message: string; timestamp: string }[]>(
    [],
  );
  const [showLogs, setShowLogs] = useState(false);
  const [logsLoading, setLogsLoading] = useState(false);
  const [liftChartSource, setLiftChartSource] = useState<"all" | "cv" | "test">(
    "all",
  );

  useEffect(() => {
    if (showLogs && logs.length === 0) {
      fetchLogs();
    }
  }, [showLogs]);

  const fetchLogs = async () => {
    setLogsLoading(true);
    try {
      const response = await fetch(
        `http://localhost:8000/api/projects/${model.id}/logs`,
      );
      if (response.ok) {
        const data = await response.json();
        setLogs(data.logs || []);
      }
    } catch (error) {
      console.error("Error fetching logs:", error);
    } finally {
      setLogsLoading(false);
    }
  };

  const featureData = Object.entries(model.feature_importance || {})
    .map(([feature, importance]) => {
      const value =
        typeof importance === "number"
          ? importance
          : parseFloat(String(importance));
      return {
        feature,
        importance: isNaN(value) ? 0 : value,
      };
    })
    .filter((item) => item.importance > 0) // Filter out zero importance
    .sort((a, b) => b.importance - a.importance)
    .slice(0, 15); // Show top 15 features

  const getStatusIcon = () => {
    if (model.status === "completed") {
      return <CheckCircle className="h-5 w-5 text-green-500" />;
    }
    return <AlertCircle className="h-5 w-5 text-yellow-500" />;
  };

  if (model.status !== "completed") {
    return (
      <div className="w-full max-w-6xl mx-auto p-6">
        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center space-x-2">
            {getStatusIcon()}
            <h2 className="text-xl font-semibold">
              Model Training in Progress...
            </h2>
          </div>
          <p className="mt-2 text-gray-600">
            Please wait while your model is being trained with parameter tuning.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="w-full max-w-6xl mx-auto p-6 space-y-6">
      <div className="bg-white rounded-lg shadow p-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-semibold">Model: {model.name}</h2>
          <div className="flex items-center space-x-2">
            {getStatusIcon()}
            <span className="text-sm text-gray-500">
              Trained at {new Date(model.created_at).toLocaleString()}
            </span>
          </div>
        </div>

        <div className="mb-6 space-y-4">
          {/* Combined Dataset & Model Info */}
          <div className="p-4 bg-gray-50 rounded-lg">
            <h3 className="text-sm font-semibold text-gray-700 mb-3">
              Dataset & Model Configuration
            </h3>
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-x-3 gap-y-2 text-sm">
              {/* Dataset Info */}
              <div className="flex items-center">
                <span className="text-gray-600">File:</span>
                <span className="ml-1 font-semibold">
                  {model.csv_filename || "Unknown"}
                </span>
              </div>
              <div className="flex items-center">
                <span className="text-gray-600">Total Rows:</span>
                <span className="ml-1 font-semibold">
                  {model.dataset_stats?.total_rows?.toLocaleString() || "N/A"}
                </span>
              </div>
              <div className="flex items-center">
                <span className="text-gray-600">Target Column:</span>
                <span className="ml-1 font-semibold">
                  {model.target_column || "Unknown"}
                </span>
              </div>
              <div className="flex items-center">
                <span className="text-gray-600">Target Rate:</span>
                <span className="ml-1 font-semibold">
                  {model.dataset_stats?.target_rate !== undefined
                    ? `${(model.dataset_stats.target_rate * 100).toFixed(1)}%`
                    : "N/A"}
                </span>
              </div>
              {/* Model Config */}
              <div className="flex items-center">
                <span className="text-gray-600">Features Used:</span>
                <span className="ml-1 font-semibold">
                  {model.dataset_stats?.n_features_used ||
                    model.feature_columns?.length ||
                    "N/A"}
                </span>
              </div>
              <div className="flex items-center">
                <span className="text-gray-600">Trees:</span>
                <span className="ml-1 font-semibold">
                  {model.n_estimators_used || "N/A"}
                </span>
              </div>
              <div className="flex items-center">
                <span className="text-gray-600">Max Depth:</span>
                <span className="ml-1 font-semibold">
                  {model.best_params?.max_depth || "N/A"}
                </span>
              </div>
              <div className="flex items-center">
                <span className="text-gray-600">Learning Rate:</span>
                <span className="ml-1 font-semibold">
                  {model.best_params?.learning_rate || "N/A"}
                </span>
              </div>
              <div className="flex items-center">
                <span className="text-gray-600">Lambda:</span>
                <span className="ml-1 font-semibold">
                  {model.best_params?.lambda || "N/A"}
                </span>
              </div>
              <div className="flex items-center">
                <span className="text-gray-600">Subsample:</span>
                <span className="ml-1 font-semibold">
                  {model.best_params?.subsample || "N/A"}
                </span>
              </div>
              <div className="flex items-center">
                <span className="text-gray-600">Col Sample:</span>
                <span className="ml-1 font-semibold">
                  {model.best_params?.colsample_bytree || "N/A"}
                </span>
              </div>
              <div className="flex items-center">
                <span className="text-gray-600">Early Stop:</span>
                <span className="ml-1 font-semibold">
                  {model.n_estimators_used
                    ? `${model.n_estimators_used} trees`
                    : "N/A"}
                </span>
              </div>
            </div>
          </div>

          {/* Performance Metrics */}
          <div className="p-4 bg-green-50 rounded-lg">
            <h3 className="text-sm font-semibold text-green-900 mb-3">
              Performance Summary
            </h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              <div>
                <span className="text-green-700">CV AUC Std:</span>
                <span className="ml-2 font-semibold text-green-900">
                  {model.cv_auc_std !== undefined
                    ? `±${(model.cv_auc_std * 100).toFixed(1)}%`
                    : "N/A"}
                </span>
              </div>
              <div>
                <span className="text-green-700">Test/CV Diff:</span>
                <span className="ml-2 font-semibold text-green-900">
                  {model.cv_auc_score &&
                  (model.test_auc_score || model.auc_score)
                    ? `${(((model.test_auc_score || model.auc_score) - model.cv_auc_score) * 100).toFixed(2)}%`
                    : "N/A"}
                </span>
              </div>
              <div>
                <span className="text-green-700">Train Size:</span>
                <span className="ml-2 font-semibold text-green-900">
                  {model.train_size?.toLocaleString() || "N/A"}
                </span>
              </div>
              <div>
                <span className="text-green-700">Test Size:</span>
                <span className="ml-2 font-semibold text-green-900">
                  {model.test_size?.toLocaleString() || "N/A"}
                </span>
              </div>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="bg-green-50 rounded-lg p-4">
            <div className="flex items-center space-x-2 mb-2">
              <TrendingUp className="h-5 w-5 text-green-600" />
              <h3 className="text-lg font-medium text-green-900">CV AUC</h3>
            </div>
            <p className="text-3xl font-bold text-green-600">
              {model.cv_auc_score
                ? (model.cv_auc_score * 100).toFixed(2)
                : "--"}
              %
            </p>
            <p className="text-sm text-green-700 mt-1">
              Cross-validation average
              {model.train_size &&
                ` (${model.train_size.toLocaleString()} rows)`}
            </p>
          </div>

          <div className="bg-blue-50 rounded-lg p-4">
            <div className="flex items-center space-x-2 mb-2">
              <TrendingUp className="h-5 w-5 text-blue-600" />
              <h3 className="text-lg font-medium text-blue-900">Test AUC</h3>
            </div>
            <p className="text-3xl font-bold text-blue-600">
              {((model.test_auc_score || model.auc_score) * 100).toFixed(2)}%
            </p>
            <p className="text-sm text-blue-700 mt-1">
              Out-of-sample performance
              {model.test_size && ` (${model.test_size.toLocaleString()} rows)`}
            </p>
          </div>
        </div>
      </div>

      {(model.lift_chart_data_multi || model.lift_chart_data) && (
        <div className="bg-white rounded-lg shadow p-6 w-full">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center space-x-2">
              <TrendingUp className="h-5 w-5 text-blue-600" />
              <h3 className="text-lg font-semibold">Model Lift Chart</h3>
            </div>
            {model.lift_chart_data_multi && (
              <div className="flex items-center space-x-2">
                <label className="text-sm text-gray-600">Data Source:</label>
                <select
                  value={liftChartSource}
                  onChange={(e) =>
                    setLiftChartSource(e.target.value as "all" | "cv" | "test")
                  }
                  className="text-sm border border-gray-300 rounded px-3 py-1 focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="all">All Data (CV + Test)</option>
                  <option value="cv">CV Data Only</option>
                  <option value="test">Test Data Only</option>
                </select>
              </div>
            )}
          </div>

          {(() => {
            let chartData = model.lift_chart_data || [];

            if (model.lift_chart_data_multi) {
              if (
                liftChartSource === "all" &&
                model.lift_chart_data_multi.all
              ) {
                chartData = model.lift_chart_data_multi.all;
              } else if (
                liftChartSource === "cv" &&
                model.lift_chart_data_multi.cv
              ) {
                chartData = model.lift_chart_data_multi.cv;
              } else if (
                liftChartSource === "test" &&
                model.lift_chart_data_multi.test
              ) {
                chartData = model.lift_chart_data_multi.test;
              }
            }

            const sourceLabel =
              liftChartSource === "all"
                ? "Combined CV + Test Data"
                : liftChartSource === "cv"
                  ? "Cross-Validation Data Only"
                  : "Test Data Only";

            return (
              <div>
                <p className="text-sm text-gray-500 mb-2">
                  Currently showing:{" "}
                  <span className="font-semibold">{sourceLabel}</span>
                </p>
                <LiftChart key={liftChartSource} data={chartData} />
              </div>
            );
          })()}
        </div>
      )}

      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold mb-4">Feature Importance</h3>
        {featureData.length > 0 ? (
          <div className="space-y-2">
            {featureData.map((item, index) => {
              const maxImportance = Math.max(
                ...featureData.map((d) => d.importance),
              );
              const percentage = (item.importance / maxImportance) * 100;

              return (
                <div
                  key={index}
                  className="grid grid-cols-12 gap-2 items-center"
                >
                  <div
                    className="col-span-4 text-xs font-medium text-gray-700 text-right break-all pr-2"
                    title={item.feature}
                  >
                    {item.feature}
                  </div>
                  <div className="col-span-8">
                    <div className="relative">
                      <div className="w-full bg-gray-200 rounded-full h-6">
                        <div
                          className="bg-blue-600 h-6 rounded-full flex items-center justify-end pr-2"
                          style={{ width: `${percentage}%` }}
                        >
                          <span className="text-xs text-white font-medium">
                            {item.importance.toFixed(4)}
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        ) : (
          <p className="text-gray-500 text-center py-8">
            No feature importance data available
          </p>
        )}
      </div>

      <div className="bg-white rounded-lg shadow p-6">
        <div
          className="flex items-center justify-between cursor-pointer"
          onClick={() => setShowLogs(!showLogs)}
        >
          <div className="flex items-center space-x-2">
            <FileText className="h-5 w-5 text-gray-600" />
            <h3 className="text-lg font-semibold">Training Logs</h3>
          </div>
          {showLogs ? (
            <ChevronUp className="h-5 w-5 text-gray-600" />
          ) : (
            <ChevronDown className="h-5 w-5 text-gray-600" />
          )}
        </div>

        {showLogs && (
          <div className="mt-4">
            {logsLoading ? (
              <p className="text-gray-500 text-center py-4">Loading logs...</p>
            ) : logs.length > 0 ? (
              <div className="space-y-2 max-h-96 overflow-y-auto bg-gray-50 rounded p-4">
                {logs.map((log, index) => (
                  <div
                    key={index}
                    className="text-sm border-b border-gray-200 pb-2"
                  >
                    <span className="text-gray-500 font-mono">
                      {new Date(log.timestamp).toLocaleTimeString()}
                    </span>
                    <span className="ml-3 text-gray-700">{log.message}</span>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-gray-500 text-center py-4">
                No training logs available
              </p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
