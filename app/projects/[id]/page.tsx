"use client";

import { useState, useEffect, useRef } from "react";
import { useParams } from "next/navigation";
import { ModelResults } from "@/components/model-results";
import { TrainingProgress } from "@/components/training-progress";
import Link from "next/link";

export default function ProjectView() {
  const params = useParams();
  const projectId = params.id as string;
  const [project, setProject] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showCode, setShowCode] = useState(false);
  const [generatedCode, setGeneratedCode] = useState<string>("");
  const [codeLoading, setCodeLoading] = useState(false);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    if (projectId) {
      fetchProject();

      // Set up polling interval
      intervalRef.current = setInterval(fetchProject, 5000); // Poll for updates if still training
    }

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [projectId]);

  const fetchProject = async () => {
    try {
      const response = await fetch(
        `http://localhost:8000/api/projects/${projectId}`,
      );
      if (response.ok) {
        const data = await response.json();
        setProject(data);

        // Stop polling if project is completed or failed
        if (data.status === "completed" || data.status === "failed") {
          setLoading(false);

          // Clear the polling interval
          if (intervalRef.current) {
            clearInterval(intervalRef.current);
            intervalRef.current = null;
          }
        }
      } else {
        setError("Project not found");
        setLoading(false);

        // Clear interval on error
        if (intervalRef.current) {
          clearInterval(intervalRef.current);
          intervalRef.current = null;
        }
      }
    } catch (error) {
      console.error("Failed to fetch project:", error);
      setError("Failed to load project");
      setLoading(false);

      // Clear interval on error
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    }
  };

  const handleGenerateCode = async () => {
    setCodeLoading(true);
    try {
      const response = await fetch(
        `http://localhost:8000/api/projects/${projectId}/generate-code`,
      );
      if (response.ok) {
        const data = await response.json();
        setGeneratedCode(data.code);
        setShowCode(true);
      } else {
        alert("Failed to generate code");
      }
    } catch (error) {
      console.error("Failed to generate code:", error);
      alert("Failed to generate code");
    } finally {
      setCodeLoading(false);
    }
  };

  const handleCopyCode = () => {
    navigator.clipboard.writeText(generatedCode);
    alert("Code copied to clipboard!");
  };

  if (loading && !project) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="text-lg">Loading project...</div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gray-50">
        <header className="bg-white shadow">
          <div className="max-w-7xl mx-auto py-6 px-4 sm:px-6 lg:px-8">
            <div className="flex items-center justify-between">
              <h1 className="text-3xl font-bold text-gray-900">Error</h1>
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
            <div className="text-center text-red-600">{error}</div>
          </div>
        </main>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white shadow">
        <div className="max-w-7xl mx-auto py-6 px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between">
            <h1 className="text-3xl font-bold text-gray-900">
              {project?.name || "XGBoost Project"}
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
          {project?.status === "training" && (
            <TrainingProgress modelId={projectId} />
          )}

          {project?.status === "completed" && (
            <>
              <ModelResults model={project} />
              <div className="mt-6 text-center space-x-4">
                <Link href="/">
                  <button className="bg-gray-600 text-white py-2 px-6 rounded-md hover:bg-gray-700">
                    Back to Projects
                  </button>
                </Link>
                <button
                  onClick={handleGenerateCode}
                  disabled={codeLoading}
                  className="bg-green-600 text-white py-2 px-6 rounded-md hover:bg-green-700 disabled:opacity-50"
                >
                  {codeLoading ? "Generating..." : "Generate Python Code"}
                </button>
                <Link href="/projects/new">
                  <button className="bg-blue-600 text-white py-2 px-6 rounded-md hover:bg-blue-700">
                    Create New Project
                  </button>
                </Link>
              </div>

              {showCode && (
                <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
                  <div className="bg-white rounded-lg max-w-6xl w-full max-h-[90vh] flex flex-col">
                    <div className="p-4 border-b">
                      <h2 className="text-xl font-semibold">
                        Generated Python Code
                      </h2>
                      <p className="text-sm text-gray-600 mt-1">
                        This code reproduces the exact XGBoost model training
                        process
                      </p>
                    </div>
                    <div className="flex-1 overflow-auto p-4">
                      <pre className="bg-gray-100 p-4 rounded-lg text-sm font-mono overflow-x-auto">
                        <code>{generatedCode}</code>
                      </pre>
                    </div>
                    <div className="p-4 border-t flex justify-end space-x-4">
                      <button
                        onClick={() => setShowCode(false)}
                        className="px-4 py-2 text-gray-600 hover:text-gray-800"
                      >
                        Close
                      </button>
                      <button
                        onClick={handleCopyCode}
                        className="bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700"
                      >
                        Copy to Clipboard
                      </button>
                    </div>
                  </div>
                </div>
              )}
            </>
          )}

          {project?.status === "failed" && (
            <div className="bg-red-50 p-6 rounded-lg">
              <h2 className="text-xl font-semibold text-red-800 mb-2">
                Training Failed
              </h2>
              <p className="text-red-600">
                The training process failed. Please check the logs for more
                details.
              </p>
              <div className="mt-4 space-x-4">
                <Link href="/">
                  <button className="bg-gray-600 text-white py-2 px-4 rounded-md hover:bg-gray-700">
                    Back to Projects
                  </button>
                </Link>
                <Link href="/projects/new">
                  <button className="bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700">
                    Try Again
                  </button>
                </Link>
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
