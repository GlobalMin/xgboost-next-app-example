"use client";

import React, { useState } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";

interface DataPreviewProps {
  data: {
    filename: string;
    columns: string[];
    shape: [number, number];
    dtypes: Record<string, string>;
    preview: Record<string, any>[];
  };
  onTrain: (config: {
    modelName: string;
    targetColumn: string;
    featureColumns: string[];
    params: any;
  }) => void;
}

export function DataPreview({ data, onTrain }: DataPreviewProps) {
  const [showPreview, setShowPreview] = useState(true);
  const [targetColumn, setTargetColumn] = useState("");
  const [featureColumns, setFeatureColumns] = useState<string[]>([]);
  const [featureScores, setFeatureScores] = useState<Record<string, number>>(
    {},
  );
  const [checkingSignal, setCheckingSignal] = useState(false);
  const [params, setParams] = useState({
    testSize: 0.2,
    cvFolds: 3,
    tuneParameters: true,
    earlyStoppingRounds: 50,
    customParamGrid: null as any,
  });

  // Default custom parameters when auto-tune is disabled
  const [customParams, setCustomParams] = useState({
    max_depth: "3,4,5",
    learning_rate: "0.01,0.1",
    lambda: "0,0.5,1.0",
    alpha: "0",
    min_child_weight: "1",
    subsample: "0.8,1.0",
    colsample_bytree: "0.8,1.0",
  });

  const handleFeatureToggle = (column: string) => {
    if (column === targetColumn) return;

    setFeatureColumns((prev) =>
      prev.includes(column)
        ? prev.filter((c) => c !== column)
        : [...prev, column],
    );
  };

  const handleTargetSelect = (column: string) => {
    setTargetColumn(column);
    setFeatureColumns((prev) => prev.filter((c) => c !== column));
    // Reset feature scores when target changes
    setFeatureScores({});
  };

  const checkFeatureSignal = async () => {
    if (!targetColumn) {
      alert("Please select a target column first");
      return;
    }

    setCheckingSignal(true);
    try {
      const response = await fetch(
        "http://localhost:8000/api/check-feature-signal",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            csv_filename: data.filename,
            target_column: targetColumn,
            feature_columns: data.columns.filter((col) => col !== targetColumn),
          }),
        },
      );

      if (response.ok) {
        const scores = await response.json();
        setFeatureScores(scores);

        // Auto-select features with high scores (top 75%)
        const sortedFeatures = Object.entries(scores)
          .sort(([, a], [, b]) => b - a)
          .map(([feature]) => feature);

        const topFeatures = sortedFeatures.slice(
          0,
          Math.ceil(sortedFeatures.length * 0.75),
        );
        setFeatureColumns(topFeatures);
      }
    } catch (error) {
      console.error("Failed to check feature signal:", error);
      alert("Failed to check feature signal");
    } finally {
      setCheckingSignal(false);
    }
  };

  const handleTrain = () => {
    if (!targetColumn || featureColumns.length === 0) {
      alert("Please select a target column and at least one feature");
      return;
    }

    // Auto-generate model name based on dataset and timestamp
    const timestamp = new Date()
      .toISOString()
      .slice(0, 19)
      .replace(/[-:]/g, "")
      .replace("T", "_");
    const datasetName = data.filename.replace(".csv", "");
    const modelName = `${datasetName}_${targetColumn}_${timestamp}`;

    // Prepare custom parameter grid if auto-tune is disabled
    let finalParams = { ...params };
    if (!params.tuneParameters) {
      // Parse comma-separated values into arrays
      const parseParamValues = (value: string, isInt: boolean = false) => {
        return value
          .split(",")
          .map((v) => {
            const trimmed = v.trim();
            return isInt ? parseInt(trimmed) : parseFloat(trimmed);
          })
          .filter((v) => !isNaN(v));
      };

      finalParams.customParamGrid = {
        max_depth: parseParamValues(customParams.max_depth, true),
        learning_rate: parseParamValues(customParams.learning_rate),
        lambda: parseParamValues(customParams.lambda),
        alpha: parseParamValues(customParams.alpha),
        min_child_weight: parseParamValues(customParams.min_child_weight),
        subsample: parseParamValues(customParams.subsample),
        colsample_bytree: parseParamValues(customParams.colsample_bytree),
      };
    }

    onTrain({
      modelName,
      targetColumn,
      featureColumns,
      params: finalParams,
    });
  };

  return (
    <div className="w-full max-w-6xl mx-auto p-6 space-y-6">
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-xl font-semibold mb-4">Dataset Information</h2>
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <span className="font-medium">Filename:</span> {data.filename}
          </div>
          <div>
            <span className="font-medium">Shape:</span> {data.shape[0]} rows ×{" "}
            {data.shape[1]} columns
          </div>
        </div>
      </div>

      <div className="bg-white rounded-lg shadow p-6">
        <button
          onClick={() => setShowPreview(!showPreview)}
          className="flex items-center space-x-2 text-lg font-semibold mb-4"
        >
          {showPreview ? (
            <ChevronDown className="h-5 w-5" />
          ) : (
            <ChevronRight className="h-5 w-5" />
          )}
          <span>Data Preview</span>
        </button>

        {showPreview && (
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  {data.columns.map((column) => (
                    <th
                      key={column}
                      className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                    >
                      {column}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {data.preview.slice(0, 5).map((row, idx) => (
                  <tr key={idx}>
                    {data.columns.map((column) => (
                      <td
                        key={column}
                        className="px-6 py-4 whitespace-nowrap text-sm text-gray-900"
                      >
                        {row[column]?.toString() || "-"}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-xl font-semibold mb-4">Model Configuration</h2>

        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Target Column
            </label>
            <select
              value={targetColumn}
              onChange={(e) => handleTargetSelect(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-blue-500 focus:border-blue-500"
            >
              <option value="">Select target column</option>
              {data.columns.map((column) => (
                <option key={column} value={column}>
                  {column} ({data.dtypes[column]})
                </option>
              ))}
            </select>
          </div>

          {targetColumn && (
            <div className="flex justify-end">
              <button
                onClick={checkFeatureSignal}
                disabled={checkingSignal}
                className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {checkingSignal ? "Checking..." : "Check Features for Signal"}
              </button>
            </div>
          )}

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Feature Columns{" "}
              {Object.keys(featureScores).length > 0 &&
                "(sorted by signal strength)"}
            </label>
            <div className="space-y-2 max-h-96 overflow-y-auto border border-gray-200 rounded-md p-3">
              {(() => {
                // Sort columns by feature scores if available
                const sortedColumns =
                  Object.keys(featureScores).length > 0
                    ? [...data.columns].sort((a, b) => {
                        const scoreA = featureScores[a] || 0;
                        const scoreB = featureScores[b] || 0;
                        return scoreB - scoreA;
                      })
                    : data.columns;

                return sortedColumns.map((column) => {
                  const score = featureScores[column];
                  const maxScore = Math.max(...Object.values(featureScores));
                  const normalizedScore =
                    score && maxScore > 0 ? score / maxScore : 0;

                  return (
                    <label
                      key={column}
                      className={`flex items-center justify-between p-2 rounded ${
                        column === targetColumn
                          ? "opacity-50 cursor-not-allowed bg-gray-50"
                          : "cursor-pointer hover:bg-gray-50"
                      }`}
                    >
                      <div className="flex items-center space-x-2">
                        <input
                          type="checkbox"
                          checked={featureColumns.includes(column)}
                          onChange={() => handleFeatureToggle(column)}
                          disabled={column === targetColumn}
                          className="h-4 w-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
                        />
                        <span className="text-sm">
                          {column} ({data.dtypes[column]})
                        </span>
                      </div>
                      {score !== undefined && (
                        <div className="flex items-center space-x-2">
                          <div className="w-24 bg-gray-200 rounded-full h-2">
                            <div
                              className="bg-blue-600 h-2 rounded-full"
                              style={{ width: `${normalizedScore * 100}%` }}
                            />
                          </div>
                          <span className="text-xs text-gray-600 w-12 text-right">
                            {score.toFixed(1)}
                          </span>
                        </div>
                      )}
                    </label>
                  );
                });
              })()}
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Test Size
              </label>
              <input
                type="number"
                value={params.testSize}
                onChange={(e) =>
                  setParams({ ...params, testSize: parseFloat(e.target.value) })
                }
                min="0.1"
                max="0.5"
                step="0.1"
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-blue-500 focus:border-blue-500"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                CV Folds
              </label>
              <input
                type="number"
                value={params.cvFolds}
                onChange={(e) =>
                  setParams({ ...params, cvFolds: parseInt(e.target.value) })
                }
                min="2"
                max="10"
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-blue-500 focus:border-blue-500"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Early Stopping Rounds
              </label>
              <input
                type="number"
                value={params.earlyStoppingRounds}
                onChange={(e) =>
                  setParams({
                    ...params,
                    earlyStoppingRounds: parseInt(e.target.value),
                  })
                }
                min="10"
                max="200"
                step="10"
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-blue-500 focus:border-blue-500"
              />
            </div>
            <div className="flex items-center">
              <label className="flex items-center space-x-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={params.tuneParameters}
                  onChange={(e) =>
                    setParams({ ...params, tuneParameters: e.target.checked })
                  }
                  className="h-4 w-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
                />
                <span className="text-sm font-medium text-gray-700">
                  Auto-tune Parameters
                </span>
              </label>
            </div>
          </div>

          {!params.tuneParameters && (
            <div className="mt-4 p-4 bg-gray-50 rounded-md space-y-3">
              <h4 className="text-sm font-semibold text-gray-700">
                Custom Parameter Grid
              </h4>
              <p className="text-xs text-gray-600 mb-2">
                Enter comma-separated values to test multiple options (e.g.,
                3,4,5)
              </p>
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Max Depth
                  </label>
                  <input
                    type="text"
                    value={customParams.max_depth}
                    onChange={(e) =>
                      setCustomParams({
                        ...customParams,
                        max_depth: e.target.value,
                      })
                    }
                    placeholder="e.g., 3,4,5"
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Learning Rate
                  </label>
                  <input
                    type="text"
                    value={customParams.learning_rate}
                    onChange={(e) =>
                      setCustomParams({
                        ...customParams,
                        learning_rate: e.target.value,
                      })
                    }
                    placeholder="e.g., 0.01,0.05,0.1"
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Lambda (L2)
                  </label>
                  <input
                    type="text"
                    value={customParams.lambda}
                    onChange={(e) =>
                      setCustomParams({
                        ...customParams,
                        lambda: e.target.value,
                      })
                    }
                    placeholder="e.g., 0,0.5,1"
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Alpha (L1)
                  </label>
                  <input
                    type="text"
                    value={customParams.alpha}
                    onChange={(e) =>
                      setCustomParams({
                        ...customParams,
                        alpha: e.target.value,
                      })
                    }
                    placeholder="e.g., 0,0.1,0.5"
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Min Child Weight
                  </label>
                  <input
                    type="text"
                    value={customParams.min_child_weight}
                    onChange={(e) =>
                      setCustomParams({
                        ...customParams,
                        min_child_weight: e.target.value,
                      })
                    }
                    placeholder="e.g., 1,3,5"
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Subsample
                  </label>
                  <input
                    type="text"
                    value={customParams.subsample}
                    onChange={(e) =>
                      setCustomParams({
                        ...customParams,
                        subsample: e.target.value,
                      })
                    }
                    placeholder="e.g., 0.6,0.8,1.0"
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Col Sample by Tree
                  </label>
                  <input
                    type="text"
                    value={customParams.colsample_bytree}
                    onChange={(e) =>
                      setCustomParams({
                        ...customParams,
                        colsample_bytree: e.target.value,
                      })
                    }
                    placeholder="e.g., 0.6,0.8,1.0"
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                  />
                </div>
              </div>
            </div>
          )}

          <button
            onClick={handleTrain}
            disabled={!targetColumn || featureColumns.length === 0}
            className="w-full bg-green-600 text-white py-2 px-4 rounded-md hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Train Model
          </button>
        </div>
      </div>
    </div>
  );
}
