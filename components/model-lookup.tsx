"use client";

import { useState } from "react";
import { Search } from "lucide-react";

interface ModelLookupProps {
  onModelFound: (model: any) => void;
}

export function ModelLookup({ onModelFound }: ModelLookupProps) {
  const [modelId, setModelId] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleLookup = async () => {
    if (!modelId.trim()) {
      setError("Please enter a model ID");
      return;
    }

    setLoading(true);
    setError("");

    try {
      const response = await fetch(
        `http://localhost:8000/api/models/${modelId}`,
      );

      if (!response.ok) {
        if (response.status === 404) {
          setError("Model not found");
        } else {
          setError("Failed to fetch model");
        }
        return;
      }

      const model = await response.json();

      if (model.status !== "completed") {
        setError("Model training is not completed yet");
        return;
      }

      onModelFound(model);
    } catch (err) {
      console.error("Lookup error:", err);
      setError("Network error. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      handleLookup();
    }
  };

  return (
    <div className="bg-white p-6 rounded-lg shadow">
      <h2 className="text-xl font-semibold mb-4">
        View Previous Model Results
      </h2>
      <p className="text-gray-600 mb-4">
        Enter a model ID to view its training results and performance metrics.
      </p>

      <div className="flex space-x-4">
        <input
          type="text"
          value={modelId}
          onChange={(e) => setModelId(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Enter model ID (e.g., 1)"
          className="flex-1 px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          disabled={loading}
        />
        <button
          onClick={handleLookup}
          disabled={loading || !modelId.trim()}
          className="px-6 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed flex items-center space-x-2"
        >
          <Search className="h-4 w-4" />
          <span>{loading ? "Loading..." : "View Results"}</span>
        </button>
      </div>

      {error && <p className="mt-2 text-sm text-red-600">{error}</p>}
    </div>
  );
}
