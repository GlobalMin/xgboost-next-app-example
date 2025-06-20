"use client";

import React, { useEffect, useState } from "react";
import { Loader2, CheckCircle } from "lucide-react";

interface TrainingProgressProps {
  modelId: string | null;
}

export function TrainingProgress({ modelId }: TrainingProgressProps) {
  const [progress, setProgress] = useState(5);
  const [currentStep, setCurrentStep] = useState("Initializing training...");
  const [status, setStatus] = useState("training");

  useEffect(() => {
    if (!modelId) return;

    // Poll for status updates every 2 seconds
    const interval = setInterval(async () => {
      try {
        const response = await fetch(
          `http://localhost:8000/api/projects/${modelId}`,
        );
        const data = await response.json();

        setStatus(data.status);

        // Update progress and step based on actual status
        if (data.status === "Loading and processing data") {
          setCurrentStep("Loading and processing data...");
          setProgress(15);
        } else if (data.status === "Tuning xgboost for best hyperparameters") {
          setCurrentStep(
            "Optimizing hyperparameters (this may take 20-40 seconds)...",
          );
          setProgress(50);
        } else if (data.status === "Training xgboost model") {
          setCurrentStep("Training XGBoost model...");
          setProgress(50);
        } else if (data.status === "Finalizing model and calculating metrics") {
          setCurrentStep("Finalizing model and calculating metrics...");
          setProgress(90);
        } else if (data.status === "training") {
          // Fallback for generic training status
          const elapsed = Date.now() - new Date(data.created_at).getTime();
          const estimatedDuration = 45000; // 45 seconds
          const estimatedProgress = Math.min(
            95,
            (elapsed / estimatedDuration) * 100,
          );
          setProgress(estimatedProgress);
          setCurrentStep("Training in progress...");
        } else if (data.status === "completed") {
          setProgress(100);
          setCurrentStep("Training completed!");
          clearInterval(interval);
        } else if (data.status === "failed") {
          setCurrentStep(
            "Training failed. Please check your data and try again.",
          );
          clearInterval(interval);
        } else {
          // For any other status, keep showing as training
          setStatus("training");
        }
      } catch (error) {
        console.error("Failed to fetch training status:", error);
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [modelId]);

  return (
    <div className="w-full max-w-4xl mx-auto p-6">
      <div className="bg-white rounded-lg shadow-lg p-8">
        <div className="text-center mb-8">
          {status === "completed" ? (
            <>
              <CheckCircle className="h-16 w-16 text-green-600 mx-auto mb-4" />
              <h2 className="text-2xl font-bold text-gray-900 mb-2">
                Training Complete!
              </h2>
              <p className="text-gray-600">
                Your model has been successfully trained.
              </p>
            </>
          ) : status === "failed" ? (
            <>
              <div className="h-16 w-16 rounded-full bg-red-100 flex items-center justify-center mx-auto mb-4">
                <span className="text-3xl">⚠️</span>
              </div>
              <h2 className="text-2xl font-bold text-gray-900 mb-2">
                Training Failed
              </h2>
              <p className="text-gray-600">
                Please check your data and try again.
              </p>
            </>
          ) : (
            <>
              <Loader2 className="h-16 w-16 animate-spin text-blue-600 mx-auto mb-4" />
              <h2 className="text-2xl font-bold text-gray-900 mb-2">
                Training Your Model
              </h2>
              <p className="text-gray-600">
                This typically takes 30-60 seconds. Please wait...
              </p>
            </>
          )}
        </div>

        {status !== "completed" && status !== "failed" && (
          <>
            {/* Progress Bar */}
            <div className="mb-6">
              <div className="flex justify-between text-sm text-gray-600 mb-2">
                <span>Progress</span>
                <span>{Math.round(progress)}%</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-3">
                <div
                  className="bg-blue-600 h-3 rounded-full transition-all duration-500 ease-out"
                  style={{ width: `${progress}%` }}
                />
              </div>
            </div>

            {/* Current Step */}
            <div className="text-center">
              <p className="text-gray-700 font-medium">{currentStep}</p>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
