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
    test_auc_score?: number;
    feature_importance: Record<string, number>;
    best_params?: Record<string, any>;
    n_estimators_used?: number;
    lift_chart_data?: any[];
    status: string;
  };
}

export function ModelResults({ model }: ModelResultsProps) {
  const [logs, setLogs] = useState<{ message: string; timestamp: string }[]>(
    [],
  );
  const [showLogs, setShowLogs] = useState(false);
  const [logsLoading, setLogsLoading] = useState(false);

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

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
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
            </p>
          </div>

          {model.cv_auc_score !== undefined && (
            <div className="bg-green-50 rounded-lg p-4">
              <div className="flex items-center space-x-2 mb-2">
                <TrendingUp className="h-5 w-5 text-green-600" />
                <h3 className="text-lg font-medium text-green-900">CV AUC</h3>
              </div>
              <p className="text-3xl font-bold text-green-600">
                {(model.cv_auc_score * 100).toFixed(2)}%
              </p>
              <p className="text-sm text-green-700 mt-1">
                Cross-validation average
              </p>
            </div>
          )}

          {model.n_estimators_used !== undefined && (
            <div className="bg-purple-50 rounded-lg p-4">
              <h3 className="text-lg font-medium text-purple-900 mb-2">
                Trees Used
              </h3>
              <p className="text-3xl font-bold text-purple-600">
                {model.n_estimators_used}
              </p>
              <p className="text-sm text-purple-700 mt-1">
                Stopped at iteration {model.n_estimators_used}
              </p>
            </div>
          )}
        </div>
      </div>

      {model.lift_chart_data && model.lift_chart_data.length > 0 && (
        <LiftChart data={model.lift_chart_data} />
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

      {model.best_params && (
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold mb-4">Best Parameters</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {Object.entries(model.best_params).map(([param, value]) => (
              <div key={param} className="bg-gray-50 rounded p-3">
                <p className="text-sm font-medium text-gray-600">
                  {param
                    .replace(/_/g, " ")
                    .replace(/\b\w/g, (l) => l.toUpperCase())}
                </p>
                <p className="text-lg font-semibold text-gray-900">
                  {typeof value === "number" ? value.toFixed(3) : value}
                </p>
              </div>
            ))}
          </div>
        </div>
      )}

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
