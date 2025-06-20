"use client";

import { useState } from "react";
import { CSVUpload } from "@/components/csv-upload";
import { DataPreview } from "@/components/data-preview";
import { ModelResults } from "@/components/model-results";
import { TrainingProgress } from "@/components/training-progress";
import { ModelLookup } from "@/components/model-lookup";
import { RecentModels } from "@/components/recent-models";

export default function Home() {
  const [uploadedData, setUploadedData] = useState<any>(null);
  const [currentModel, setCurrentModel] = useState<any>(null);
  const [training, setTraining] = useState(false);
  const [trainingModelId, setTrainingModelId] = useState<number | null>(null);

  const handleUpload = (data: any) => {
    setUploadedData(data);
    setCurrentModel(null);
  };

  const handleTrain = async (config: {
    modelName: string;
    targetColumn: string;
    featureColumns: string[];
    params: any;
  }) => {
    setTraining(true);

    try {
      const response = await fetch("http://localhost:8000/api/train", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          model_name: config.modelName,
          csv_filename: uploadedData.filename,
          target_column: config.targetColumn,
          feature_columns: config.featureColumns,
          test_size: config.params.testSize,
          cv_folds: config.params.cvFolds,
          tune_parameters: config.params.tuneParameters,
          early_stopping_rounds: config.params.earlyStoppingRounds,
        }),
      });

      if (!response.ok) {
        throw new Error("Training failed");
      }

      const result = await response.json();
      setTrainingModelId(result.model_id);

      // Poll for completion
      const checkStatus = setInterval(async () => {
        const statusResponse = await fetch(
          `http://localhost:8000/api/models/${result.model_id}`,
        );
        if (statusResponse.ok) {
          const modelData = await statusResponse.json();
          if (modelData.status === "completed") {
            clearInterval(checkStatus);
            // Add a 1-second pause before showing results
            setTimeout(() => {
              setCurrentModel({
                ...modelData,
                ...result,
                name: config.modelName,
              });
              setTraining(false);
              setTrainingModelId(null);
            }, 1000);
          } else if (modelData.status === "failed") {
            setTraining(false);
            setTrainingModelId(null);
            clearInterval(checkStatus);
            alert("Training failed. Check the logs for details.");
          }
        }
      }, 2000);
    } catch (error) {
      console.error("Training error:", error);
      alert("Training failed. Please check the console for details.");
      setTraining(false);
      setTrainingModelId(null);
    }
  };

  const handleModelLookup = async (modelId: number) => {
    try {
      const response = await fetch(
        `http://localhost:8000/api/models/${modelId}`,
      );
      if (response.ok) {
        const model = await response.json();
        setCurrentModel(model);
        setUploadedData(null);
      }
    } catch (error) {
      console.error("Failed to fetch model:", error);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white shadow">
        <div className="max-w-7xl mx-auto py-6 px-4 sm:px-6 lg:px-8">
          <h1 className="text-3xl font-bold text-gray-900">
            XGBoost Model Trainer
          </h1>
        </div>
      </header>

      <main className="py-10">
        <div className="max-w-7xl mx-auto sm:px-6 lg:px-8">
          {!uploadedData && !currentModel && !training && (
            <div className="space-y-6">
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <CSVUpload onUpload={handleUpload} />
                <ModelLookup onModelFound={setCurrentModel} />
              </div>
              <RecentModels onSelectModel={handleModelLookup} />
            </div>
          )}

          {uploadedData && !currentModel && !training && (
            <DataPreview data={uploadedData} onTrain={handleTrain} />
          )}

          {training && <TrainingProgress modelId={trainingModelId} />}

          {currentModel && (
            <>
              <ModelResults model={currentModel} />
              <div className="mt-6 text-center space-x-4">
                <button
                  onClick={() => {
                    setUploadedData(null);
                    setCurrentModel(null);
                  }}
                  className="bg-gray-600 text-white py-2 px-6 rounded-md hover:bg-gray-700"
                >
                  Back to Home
                </button>
                <button
                  onClick={() => {
                    setUploadedData(null);
                    setCurrentModel(null);
                  }}
                  className="bg-blue-600 text-white py-2 px-6 rounded-md hover:bg-blue-700"
                >
                  Train Another Model
                </button>
              </div>
            </>
          )}
        </div>
      </main>
    </div>
  );
}
