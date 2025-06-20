"use client";

import { useState, useEffect } from "react";
import { Clock, TrendingUp, AlertCircle, CheckCircle } from "lucide-react";

interface RecentModelsProps {
  onSelectModel: (modelId: number) => void;
}

export function RecentModels({ onSelectModel }: RecentModelsProps) {
  const [models, setModels] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchModels();
  }, []);

  const fetchModels = async () => {
    try {
      const response = await fetch("http://localhost:8000/api/models");
      if (response.ok) {
        const data = await response.json();
        setModels(data.slice(0, 5)); // Show only 5 most recent
      }
    } catch (error) {
      console.error("Failed to fetch models:", error);
    } finally {
      setLoading(false);
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "completed":
        return <CheckCircle className="h-4 w-4 text-green-500" />;
      case "failed":
        return <AlertCircle className="h-4 w-4 text-red-500" />;
      default:
        return <Clock className="h-4 w-4 text-yellow-500" />;
    }
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString() + " " + date.toLocaleTimeString();
  };

  if (loading) {
    return (
      <div className="bg-white p-6 rounded-lg shadow">
        <h3 className="text-lg font-semibold mb-4">Recent Models</h3>
        <div className="text-gray-500">Loading...</div>
      </div>
    );
  }

  if (models.length === 0) {
    return null;
  }

  return (
    <div className="bg-white p-6 rounded-lg shadow">
      <h3 className="text-lg font-semibold mb-4">Recent Models</h3>
      <div className="space-y-3">
        {models.map((model) => (
          <div
            key={model.id}
            className="border rounded-lg p-3 hover:bg-gray-50 cursor-pointer transition-colors"
            onClick={() =>
              model.status === "completed" && onSelectModel(model.id)
            }
          >
            <div className="flex items-start justify-between">
              <div className="flex-1">
                <div className="flex items-center space-x-2">
                  <span className="font-medium">#{model.id}</span>
                  <span className="text-gray-700">{model.name}</span>
                  {getStatusIcon(model.status)}
                </div>
                <div className="text-sm text-gray-500 mt-1">
                  {formatDate(model.created_at)}
                </div>
                {model.auc_score && (
                  <div className="flex items-center space-x-1 mt-1">
                    <TrendingUp className="h-3 w-3 text-blue-500" />
                    <span className="text-sm text-gray-600">
                      AUC: {model.auc_score.toFixed(4)}
                    </span>
                  </div>
                )}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
