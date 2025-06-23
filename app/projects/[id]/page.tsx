"use client";

import { useState, useEffect, useRef } from "react";
import { useParams } from "next/navigation";
import { ModelResults } from "@/components/model-results";
import { TrainingProgress } from "@/components/training-progress";
import Link from "next/link";
import { ArrowLeft, Code2, Plus, Home, Copy, X } from "lucide-react";

export default function ProjectView() {
  const params = useParams();
  const projectId = params.id as string;
  const [project, setProject] = useState<Record<string, any> | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showCode, setShowCode] = useState(false);
  const [generatedCode, setGeneratedCode] = useState<string>("");
  const [codeLoading, setCodeLoading] = useState(false);
  const [copySuccess, setCopySuccess] = useState(false);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    const fetchProjectData = async () => {
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

    if (projectId) {
      fetchProjectData();

      // Set up polling interval
      intervalRef.current = setInterval(fetchProjectData, 5000); // Poll for updates if still training
    }

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [projectId]);


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
    setCopySuccess(true);
    setTimeout(() => setCopySuccess(false), 2000);
  };

  if (loading && !project) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <div className="text-lg text-gray-600">Loading project...</div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gray-50">
        <header className="bg-gradient-to-r from-red-500 to-red-600 text-white shadow-lg">
          <div className="max-w-7xl mx-auto py-6 px-4 sm:px-6 lg:px-8">
            <div className="flex items-center justify-between">
              <h1 className="text-3xl font-bold">Error</h1>
              <Link href="/">
                <button className="flex items-center gap-2 bg-white/10 hover:bg-white/20 px-4 py-2 rounded-lg transition-colors backdrop-blur-sm">
                  <ArrowLeft className="h-4 w-4" />
                  Back to Projects
                </button>
              </Link>
            </div>
          </div>
        </header>
        <main className="py-10">
          <div className="max-w-7xl mx-auto sm:px-6 lg:px-8">
            <div className="bg-white rounded-lg shadow-sm p-8 text-center">
              <div className="text-red-600 text-xl mb-4">{error}</div>
              <Link href="/">
                <button className="bg-gray-100 hover:bg-gray-200 text-gray-700 px-6 py-2 rounded-lg transition-colors">
                  Return to Projects
                </button>
              </Link>
            </div>
          </div>
        </main>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-gradient-to-r from-blue-600 to-blue-700 text-white shadow-lg">
        <div className="max-w-7xl mx-auto py-6 px-4 sm:px-6 lg:px-8">
          <nav className="flex items-center text-sm mb-4">
            <Link href="/" className="flex items-center hover:text-blue-100 transition-colors">
              <Home className="h-4 w-4 mr-1" />
              Projects
            </Link>
            <span className="mx-2 text-blue-200">/</span>
            <span className="text-blue-100">{project?.name || "Loading..."}</span>
          </nav>
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold">
                {project?.name || "XGBoost Project"}
              </h1>
              {project?.created_at && (
                <p className="text-blue-100 mt-1">
                  Created {new Date(project.created_at).toLocaleDateString()}
                </p>
              )}
            </div>
            <Link href="/">
              <button className="flex items-center gap-2 bg-white/10 hover:bg-white/20 px-4 py-2 rounded-lg transition-colors backdrop-blur-sm">
                <ArrowLeft className="h-4 w-4" />
                Back to Projects
              </button>
            </Link>
          </div>
        </div>
      </header>

      <main className="py-8">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          {project?.status === "training" && (
            <TrainingProgress modelId={projectId} />
          )}

          {project?.status === "completed" && (
            <>
              <ModelResults model={project} />
              <div className="mt-8 bg-white rounded-lg shadow-sm p-6">
                <div className="flex flex-wrap gap-4 justify-center">
                  <Link href="/">
                    <button className="flex items-center gap-2 bg-gray-100 hover:bg-gray-200 text-gray-700 py-3 px-6 rounded-lg transition-all hover:shadow-md">
                      <ArrowLeft className="h-4 w-4" />
                      Back to Projects
                    </button>
                  </Link>
                  <button
                    onClick={handleGenerateCode}
                    disabled={codeLoading}
                    className="flex items-center gap-2 bg-gradient-to-r from-green-500 to-green-600 hover:from-green-600 hover:to-green-700 text-white py-3 px-6 rounded-lg transition-all hover:shadow-md disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    <Code2 className="h-4 w-4" />
                    {codeLoading ? "Generating..." : "Generate Python Code"}
                  </button>
                  <Link href="/projects/new">
                    <button className="flex items-center gap-2 bg-gradient-to-r from-blue-500 to-blue-600 hover:from-blue-600 hover:to-blue-700 text-white py-3 px-6 rounded-lg transition-all hover:shadow-md">
                      <Plus className="h-4 w-4" />
                      Create New Project
                    </button>
                  </Link>
                </div>
              </div>

              {showCode && (
                <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 p-4 animate-fadeIn">
                  <div className="bg-white rounded-xl max-w-6xl w-full max-h-[90vh] flex flex-col shadow-2xl animate-slideUp">
                    <div className="p-6 border-b bg-gradient-to-r from-gray-50 to-gray-100">
                      <div className="flex items-center justify-between">
                        <div>
                          <h2 className="text-2xl font-semibold text-gray-900">
                            Generated Python Code
                          </h2>
                          <p className="text-sm text-gray-600 mt-1">
                            Complete XGBoost model training pipeline
                          </p>
                        </div>
                        <button
                          onClick={() => setShowCode(false)}
                          className="p-2 hover:bg-gray-200 rounded-lg transition-colors"
                        >
                          <X className="h-5 w-5 text-gray-500" />
                        </button>
                      </div>
                    </div>
                    <div className="flex-1 overflow-auto p-6 bg-gray-50">
                      <div className="bg-gray-900 p-6 rounded-lg shadow-inner">
                        <pre className="text-sm font-mono overflow-x-auto">
                          <code className="text-gray-100">{generatedCode}</code>
                        </pre>
                      </div>
                    </div>
                    <div className="p-6 border-t bg-gray-50 flex justify-end gap-4">
                      <button
                        onClick={() => setShowCode(false)}
                        className="px-6 py-2.5 text-gray-600 hover:text-gray-800 hover:bg-gray-100 rounded-lg transition-colors"
                      >
                        Close
                      </button>
                      <button
                        onClick={handleCopyCode}
                        className="bg-gradient-to-r from-blue-500 to-blue-600 hover:from-blue-600 hover:to-blue-700 text-white px-6 py-2.5 rounded-lg transition-all hover:shadow-md flex items-center gap-2"
                      >
                        <Copy className="h-4 w-4" />
                        {copySuccess ? "Copied!" : "Copy to Clipboard"}
                      </button>
                    </div>
                  </div>
                </div>
              )}
            </>
          )}

          {project?.status === "failed" && (
            <div className="bg-red-50 border border-red-200 p-6 rounded-lg">
              <h2 className="text-xl font-semibold text-red-800 mb-2">
                Training Failed
              </h2>
              <p className="text-red-600">
                The training process failed. Please check the logs for more
                details.
              </p>
              <div className="mt-6 flex gap-4">
                <Link href="/">
                  <button className="bg-gray-600 hover:bg-gray-700 text-white py-2 px-6 rounded-lg transition-colors">
                    Back to Projects
                  </button>
                </Link>
                <Link href="/projects/new">
                  <button className="bg-blue-600 hover:bg-blue-700 text-white py-2 px-6 rounded-lg transition-colors">
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