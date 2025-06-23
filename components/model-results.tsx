"use client";

import React, { useState, useEffect } from "react";
import {
  CheckCircle,
  AlertCircle,
  TrendingUp,
  FileText,
  ChevronDown,
  ChevronUp,
  Database,
  Settings,
  BarChart3,
  Activity,
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
    best_params?: Record<string, number | string>;
    n_estimators_used?: number;
    lift_chart_data?: Array<{
      bin: number;
      avg_prediction: number;
      actual_rate: number;
      count: number;
    }>;
    lift_chart_data_multi?: {
      all?: Array<{
        bin: number;
        avg_prediction: number;
        actual_rate: number;
        count: number;
      }>;
      cv?: Array<{
        bin: number;
        avg_prediction: number;
        actual_rate: number;
        count: number;
      }>;
      test?: Array<{
        bin: number;
        avg_prediction: number;
        actual_rate: number;
        count: number;
      }>;
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
    model_params?: {
      cv_folds?: number;
    };
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

  // Helper function to create circular progress
  const CircularProgress = ({ value, label, color }: { value: number; label: string; color: string }) => {
    const circumference = 2 * Math.PI * 45;
    const strokeDashoffset = circumference - (value / 100) * circumference;

    return (
      <div className="relative inline-flex items-center justify-center">
        <svg className="transform -rotate-90 w-32 h-32">
          <circle
            cx="64"
            cy="64"
            r="45"
            stroke="#e5e7eb"
            strokeWidth="10"
            fill="none"
          />
          <circle
            cx="64"
            cy="64"
            r="45"
            stroke={color}
            strokeWidth="10"
            fill="none"
            strokeDasharray={circumference}
            strokeDashoffset={strokeDashoffset}
            className="transition-all duration-1000 ease-out"
          />
        </svg>
        <div className="absolute flex flex-col items-center">
          <span className="text-2xl font-bold">{value.toFixed(1)}%</span>
          <span className="text-xs text-gray-500 mt-1">{label}</span>
        </div>
      </div>
    );
  };

  return (
    <div className="w-full space-y-6">
      <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-6">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-blue-50 rounded-lg">
              <Activity className="h-6 w-6 text-blue-600" />
            </div>
            <div>
              <h2 className="text-2xl font-semibold text-gray-900">{model.name}</h2>
              <p className="text-sm text-gray-500">
                Trained on {new Date(model.created_at).toLocaleDateString()} at {new Date(model.created_at).toLocaleTimeString()}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2 bg-green-50 px-3 py-1.5 rounded-full">
            <CheckCircle className="h-4 w-4 text-green-600" />
            <span className="text-sm font-medium text-green-700">Completed</span>
          </div>
        </div>

        {/* Performance Metrics with Visual Gauges */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
          <div className="bg-gradient-to-br from-green-50 to-green-100 rounded-xl p-6 border border-green-200">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h3 className="text-lg font-semibold text-green-900">Cross-Validation AUC</h3>
                <p className="text-sm text-green-700">Average across {model.train_size?.toLocaleString() || "all"} training samples</p>
              </div>
              <CircularProgress 
                value={model.cv_auc_score ? model.cv_auc_score * 100 : 0} 
                label="CV AUC"
                color="#10b981"
              />
            </div>
            {model.cv_auc_std && (
              <div className="mt-4 bg-white/50 rounded-lg p-3">
                <p className="text-sm text-green-800">
                  <span className="font-medium">Standard Deviation:</span> ±{(model.cv_auc_std * 100).toFixed(2)}%
                </p>
              </div>
            )}
          </div>

          <div className="bg-gradient-to-br from-blue-50 to-blue-100 rounded-xl p-6 border border-blue-200">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h3 className="text-lg font-semibold text-blue-900">Test Set AUC</h3>
                <p className="text-sm text-blue-700">Performance on {model.test_size?.toLocaleString() || "test"} samples</p>
              </div>
              <CircularProgress 
                value={(model.test_auc_score || model.auc_score) * 100} 
                label="Test AUC"
                color="#3b82f6"
              />
            </div>
            {model.cv_auc_score && (model.test_auc_score || model.auc_score) && (
              <div className="mt-4 bg-white/50 rounded-lg p-3">
                <p className="text-sm text-blue-800">
                  <span className="font-medium">vs CV Score:</span> 
                  <span className={`ml-2 font-semibold ${
                    ((model.test_auc_score || model.auc_score) - model.cv_auc_score) >= 0 ? 'text-green-700' : 'text-red-700'
                  }`}>
                    {((model.test_auc_score || model.auc_score) - model.cv_auc_score) >= 0 ? '+' : ''}
                    {(((model.test_auc_score || model.auc_score) - model.cv_auc_score) * 100).toFixed(2)}%
                  </span>
                </p>
              </div>
            )}
          </div>
        </div>

        {/* Dataset & Model Configuration Cards */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 mb-6">
          {/* Dataset Info Card */}
          <div className="bg-gradient-to-br from-gray-50 to-gray-100 rounded-xl p-5 border border-gray-200">
            <div className="flex items-center gap-2 mb-4">
              <Database className="h-5 w-5 text-gray-600" />
              <h3 className="font-semibold text-gray-800">Dataset Information</h3>
            </div>
            <div className="space-y-2.5">
              <div className="flex justify-between items-center text-sm">
                <span className="text-gray-600">File:</span>
                <span className="font-medium text-gray-900 truncate ml-2" title={model.csv_filename}>
                  {model.csv_filename || "Unknown"}
                </span>
              </div>
              <div className="flex justify-between items-center text-sm">
                <span className="text-gray-600">Total Rows:</span>
                <span className="font-medium text-gray-900">
                  {model.dataset_stats?.total_rows?.toLocaleString() || "N/A"}
                </span>
              </div>
              <div className="flex justify-between items-center text-sm">
                <span className="text-gray-600">Features:</span>
                <span className="font-medium text-gray-900">
                  {model.dataset_stats?.n_features_used || model.feature_columns?.length || "N/A"}
                </span>
              </div>
              <div className="flex justify-between items-center text-sm">
                <span className="text-gray-600">Target:</span>
                <span className="font-medium text-gray-900 truncate ml-2" title={model.target_column}>
                  {model.target_column || "Unknown"}
                </span>
              </div>
              <div className="flex justify-between items-center text-sm">
                <span className="text-gray-600">Target Rate:</span>
                <span className="font-medium text-gray-900">
                  {model.dataset_stats?.target_rate !== undefined
                    ? `${(model.dataset_stats.target_rate * 100).toFixed(1)}%`
                    : "N/A"}
                </span>
              </div>
            </div>
          </div>

          {/* Model Configuration Card */}
          <div className="bg-gradient-to-br from-purple-50 to-purple-100 rounded-xl p-5 border border-purple-200">
            <div className="flex items-center gap-2 mb-4">
              <Settings className="h-5 w-5 text-purple-600" />
              <h3 className="font-semibold text-purple-800">Model Configuration</h3>
            </div>
            <div className="space-y-2.5">
              <div className="flex justify-between items-center text-sm">
                <span className="text-purple-700">Trees:</span>
                <span className="font-medium text-purple-900">
                  {model.n_estimators_used || "N/A"}
                </span>
              </div>
              <div className="flex justify-between items-center text-sm">
                <span className="text-purple-700">Max Depth:</span>
                <span className="font-medium text-purple-900">
                  {model.best_params?.max_depth || "N/A"}
                </span>
              </div>
              <div className="flex justify-between items-center text-sm">
                <span className="text-purple-700">Learning Rate:</span>
                <span className="font-medium text-purple-900">
                  {model.best_params?.learning_rate || "N/A"}
                </span>
              </div>
              <div className="flex justify-between items-center text-sm">
                <span className="text-purple-700">Lambda:</span>
                <span className="font-medium text-purple-900">
                  {model.best_params?.lambda || "N/A"}
                </span>
              </div>
              <div className="flex justify-between items-center text-sm">
                <span className="text-purple-700">Subsample:</span>
                <span className="font-medium text-purple-900">
                  {model.best_params?.subsample || "N/A"}
                </span>
              </div>
            </div>
          </div>

          {/* Training Details Card */}
          <div className="bg-gradient-to-br from-orange-50 to-orange-100 rounded-xl p-5 border border-orange-200">
            <div className="flex items-center gap-2 mb-4">
              <BarChart3 className="h-5 w-5 text-orange-600" />
              <h3 className="font-semibold text-orange-800">Training Details</h3>
            </div>
            <div className="space-y-2.5">
              <div className="flex justify-between items-center text-sm">
                <span className="text-orange-700">Train Size:</span>
                <span className="font-medium text-orange-900">
                  {model.train_size?.toLocaleString() || "N/A"}
                </span>
              </div>
              <div className="flex justify-between items-center text-sm">
                <span className="text-orange-700">Test Size:</span>
                <span className="font-medium text-orange-900">
                  {model.test_size?.toLocaleString() || "N/A"}
                </span>
              </div>
              <div className="flex justify-between items-center text-sm">
                <span className="text-orange-700">Col Sample:</span>
                <span className="font-medium text-orange-900">
                  {model.best_params?.colsample_bytree || "N/A"}
                </span>
              </div>
              <div className="flex justify-between items-center text-sm">
                <span className="text-orange-700">Early Stop:</span>
                <span className="font-medium text-orange-900">
                  {model.n_estimators_used ? `${model.n_estimators_used} trees` : "N/A"}
                </span>
              </div>
              <div className="flex justify-between items-center text-sm">
                <span className="text-orange-700">CV Folds:</span>
                <span className="font-medium text-orange-900">{model.model_params?.cv_folds || 3}</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {(model.lift_chart_data_multi || model.lift_chart_data) && (
        <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-6">
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-blue-50 rounded-lg">
                <TrendingUp className="h-6 w-6 text-blue-600" />
              </div>
              <div>
                <h3 className="text-xl font-semibold text-gray-900">Model Lift Chart</h3>
                <p className="text-sm text-gray-500 mt-0.5">Comparison of predicted vs actual outcomes</p>
              </div>
            </div>
            {model.lift_chart_data_multi && (
              <div className="flex items-center space-x-2">
                <label className="text-sm text-gray-600">Data Source:</label>
                <select
                  value={liftChartSource}
                  onChange={(e) =>
                    setLiftChartSource(e.target.value as "all" | "cv" | "test")
                  }
                  className="text-sm border border-gray-200 rounded-lg px-4 py-2 bg-white hover:border-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-colors"
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

      <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-6">
        <div className="flex items-center gap-3 mb-6">
          <div className="p-2 bg-purple-50 rounded-lg">
            <BarChart3 className="h-6 w-6 text-purple-600" />
          </div>
          <div>
            <h3 className="text-xl font-semibold text-gray-900">Feature Importance</h3>
            <p className="text-sm text-gray-500 mt-0.5">Top {featureData.length} most influential features</p>
          </div>
        </div>
        {featureData.length > 0 ? (
          <div className="space-y-3">
            {featureData.map((item, index) => {
              const maxImportance = Math.max(
                ...featureData.map((d) => d.importance),
              );
              const percentage = (item.importance / maxImportance) * 100;
              const relativePercentage = (item.importance / featureData.reduce((sum, d) => sum + d.importance, 0)) * 100;

              return (
                <div
                  key={index}
                  className="group hover:bg-gray-50 rounded-lg p-2 -mx-2 transition-colors"
                >
                  <div className="grid grid-cols-12 gap-3 items-center">
                    <div
                      className="col-span-4 text-sm font-medium text-gray-700 text-right truncate pr-2"
                      title={item.feature}
                    >
                      {item.feature}
                    </div>
                    <div className="col-span-8">
                      <div className="relative">
                        <div className="w-full bg-gray-100 rounded-full h-7 overflow-hidden">
                          <div
                            className="bg-gradient-to-r from-purple-500 to-purple-600 h-7 rounded-full flex items-center justify-between px-3 transition-all duration-500 ease-out"
                            style={{ width: `${percentage}%` }}
                          >
                            <span className="text-xs text-white font-medium">
                              {item.importance.toFixed(4)}
                            </span>
                            {percentage > 20 && (
                              <span className="text-xs text-purple-100">
                                {relativePercentage.toFixed(1)}%
                              </span>
                            )}
                          </div>
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

      <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-6">
        <div
          className="flex items-center justify-between cursor-pointer hover:bg-gray-50 -m-2 p-2 rounded-lg transition-colors"
          onClick={() => setShowLogs(!showLogs)}
        >
          <div className="flex items-center gap-3">
            <div className="p-2 bg-gray-50 rounded-lg">
              <FileText className="h-5 w-5 text-gray-600" />
            </div>
            <div>
              <h3 className="text-lg font-semibold text-gray-900">Training Logs</h3>
              <p className="text-sm text-gray-500">View detailed training progress</p>
            </div>
          </div>
          <div className="p-2 hover:bg-gray-100 rounded-lg transition-colors">
            {showLogs ? (
              <ChevronUp className="h-5 w-5 text-gray-600" />
            ) : (
              <ChevronDown className="h-5 w-5 text-gray-600" />
            )}
          </div>
        </div>

        {showLogs && (
          <div className="mt-4">
            {logsLoading ? (
              <p className="text-gray-500 text-center py-4">Loading logs...</p>
            ) : logs.length > 0 ? (
              <div className="space-y-2 max-h-96 overflow-y-auto bg-gray-50 rounded-lg p-4 border border-gray-200">
                {logs.map((log, index) => (
                  <div
                    key={index}
                    className="text-sm border-b border-gray-100 pb-2 last:border-0"
                  >
                    <span className="text-gray-500 font-mono text-xs">
                      {new Date(log.timestamp).toLocaleTimeString()}
                    </span>
                    <span className="ml-3 text-gray-700 leading-relaxed">{log.message}</span>
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