"use client";

import { useState, useEffect } from "react";
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

  useEffect(() => {
    if (projectId) {
      fetchProject();
      const interval = setInterval(fetchProject, 5000); // Poll for updates if still training
      return () => clearInterval(interval);
    }
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
        }
      } else {
        setError("Project not found");
        setLoading(false);
      }
    } catch (error) {
      console.error("Failed to fetch project:", error);
      setError("Failed to load project");
      setLoading(false);
    }
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
                <Link href="/projects/new">
                  <button className="bg-blue-600 text-white py-2 px-6 rounded-md hover:bg-blue-700">
                    Create New Project
                  </button>
                </Link>
              </div>
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
