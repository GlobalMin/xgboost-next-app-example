"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { CSVUpload } from "@/components/csv-upload";
import { DataPreview } from "@/components/data-preview";
import { TrainingProgress } from "@/components/training-progress";
import Link from "next/link";

export default function NewProject() {
  const router = useRouter();
  const [uploadedData, setUploadedData] = useState<any>(null);
  const [training, setTraining] = useState(false);
  const [trainingProjectId, setTrainingProjectId] = useState<string | null>(
    null,
  );

  const handleUpload = (data: any) => {
    setUploadedData(data);
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
          custom_param_grid: config.params.customParamGrid,
        }),
      });

      if (!response.ok) {
        throw new Error("Training failed");
      }

      const result = await response.json();
      setTrainingProjectId(result.project_id);

      // Poll for completion
      const checkStatus = setInterval(async () => {
        const statusResponse = await fetch(
          `http://localhost:8000/api/projects/${result.project_id}`,
        );
        if (statusResponse.ok) {
          const projectData = await statusResponse.json();
          if (projectData.status === "completed") {
            clearInterval(checkStatus);
            // Redirect to project view page after training completes
            setTimeout(() => {
              router.push(`/projects/${result.project_id}`);
            }, 1000);
          } else if (projectData.status === "failed") {
            setTraining(false);
            setTrainingProjectId(null);
            clearInterval(checkStatus);
            alert("Training failed. Check the logs for details.");
          }
        }
      }, 2000);
    } catch (error) {
      console.error("Training error:", error);
      alert("Training failed. Please check the console for details.");
      setTraining(false);
      setTrainingProjectId(null);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white shadow">
        <div className="max-w-7xl mx-auto py-6 px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between">
            <h1 className="text-3xl font-bold text-gray-900">
              New XGBoost Project
            </h1>
            <Link href="/">
              <button className="text-gray-600 hover:text-gray-900">
                ← Back to Projects
              </button>
            </Link>
          </div>
        </div>
      </header>

      <main className="py-10">
        <div className="max-w-7xl mx-auto sm:px-6 lg:px-8">
          {!uploadedData && !training && <CSVUpload onUpload={handleUpload} />}

          {uploadedData && !training && (
            <DataPreview data={uploadedData} onTrain={handleTrain} />
          )}

          {training && <TrainingProgress modelId={trainingProjectId} />}
        </div>
      </main>
    </div>
  );
}
