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
  const [params, setParams] = useState({
    testSize: 0.2,
    cvFolds: 3,
    tuneParameters: true,
    earlyStoppingRounds: 50,
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

    onTrain({
      modelName,
      targetColumn,
      featureColumns,
      params,
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

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Feature Columns
            </label>
            <div className="space-y-2 max-h-48 overflow-y-auto border border-gray-200 rounded-md p-3">
              {data.columns.map((column) => (
                <label
                  key={column}
                  className={`flex items-center space-x-2 ${
                    column === targetColumn
                      ? "opacity-50 cursor-not-allowed"
                      : "cursor-pointer"
                  }`}
                >
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
                </label>
              ))}
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
