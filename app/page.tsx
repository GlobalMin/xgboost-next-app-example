"use client";

import { useState, useEffect } from "react";
import { ConfirmationModal } from "@/components/confirmation-modal";
import Link from "next/link";

export default function Home() {
  const [projects, setProjects] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [deleteModalOpen, setDeleteModalOpen] = useState(false);
  const [projectToDelete, setProjectToDelete] = useState<any>(null);
  const [isDeleting, setIsDeleting] = useState(false);

  useEffect(() => {
    fetchProjects();
  }, []);

  const fetchProjects = async () => {
    try {
      const response = await fetch("http://localhost:8000/api/projects");
      if (response.ok) {
        const data = await response.json();
        setProjects(data);
      }
    } catch (error) {
      console.error("Failed to fetch projects:", error);
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteClick = (e: React.MouseEvent, project: any) => {
    e.preventDefault();
    e.stopPropagation();
    setProjectToDelete(project);
    setDeleteModalOpen(true);
  };

  const handleDeleteConfirm = async () => {
    if (!projectToDelete) return;

    setIsDeleting(true);
    try {
      const response = await fetch(
        `http://localhost:8000/api/projects/${projectToDelete.id}`,
        {
          method: "DELETE",
        },
      );

      if (response.ok) {
        // Remove the deleted project from the list
        setProjects(projects.filter((p) => p.id !== projectToDelete.id));
      } else {
        console.error("Failed to delete project");
        alert("Failed to delete project. Please try again.");
      }
    } catch (error) {
      console.error("Error deleting project:", error);
      alert("Error deleting project. Please try again.");
    } finally {
      setIsDeleting(false);
      setProjectToDelete(null);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white shadow">
        <div className="max-w-7xl mx-auto py-6 px-4 sm:px-6 lg:px-8">
          <h1 className="text-3xl font-bold text-gray-900">XGBoost Projects</h1>
        </div>
      </header>

      <main className="py-10">
        <div className="max-w-7xl mx-auto sm:px-6 lg:px-8">
          <div className="mb-8">
            <h2 className="text-2xl font-semibold mb-4">Create New Project</h2>
            <Link href="/projects/new">
              <button className="bg-blue-600 text-white py-3 px-6 rounded-md hover:bg-blue-700">
                Start New Project
              </button>
            </Link>
          </div>

          <div>
            <h2 className="text-2xl font-semibold mb-4">Recent Projects</h2>
            {loading ? (
              <div className="text-center py-8">Loading projects...</div>
            ) : projects.length === 0 ? (
              <div className="text-center py-8 text-gray-500">
                No projects yet. Create your first project to get started!
              </div>
            ) : (
              <div className="grid gap-4">
                {projects.map((project) => (
                  <div key={project.id} className="relative group">
                    <Link href={`/projects/${project.id}`} className="block">
                      <div className="bg-white p-6 rounded-lg shadow hover:shadow-md transition-shadow">
                        <div className="flex justify-between items-start">
                          <div>
                            <h3 className="text-lg font-medium">
                              {project.name}
                            </h3>
                            <p className="text-sm text-gray-600 mt-1">
                              Dataset: {project.csv_filename}
                            </p>
                            <p className="text-sm text-gray-500 mt-1">
                              Created:{" "}
                              {new Date(
                                project.created_at,
                              ).toLocaleDateString()}
                            </p>
                          </div>
                          <div className="text-right">
                            <span
                              className={`inline-block px-3 py-1 rounded-full text-sm ${
                                project.status === "completed"
                                  ? "bg-green-100 text-green-800"
                                  : project.status === "training"
                                    ? "bg-yellow-100 text-yellow-800"
                                    : project.status === "failed"
                                      ? "bg-red-100 text-red-800"
                                      : "bg-gray-100 text-gray-800"
                              }`}
                            >
                              {project.status}
                            </span>
                            {project.auc_score && (
                              <p className="text-sm text-gray-600 mt-2">
                                AUC: {project.auc_score.toFixed(4)}
                              </p>
                            )}
                          </div>
                        </div>
                      </div>
                    </Link>
                    <button
                      onClick={(e) => handleDeleteClick(e, project)}
                      className="absolute top-2 right-2 p-2 text-gray-400 hover:text-red-600 hover:bg-red-50 rounded-md transition-colors opacity-0 group-hover:opacity-100"
                      title="Delete project"
                    >
                      <svg
                        xmlns="http://www.w3.org/2000/svg"
                        className="h-5 w-5"
                        viewBox="0 0 20 20"
                        fill="currentColor"
                      >
                        <path
                          fillRule="evenodd"
                          d="M9 2a1 1 0 00-.894.553L7.382 4H4a1 1 0 000 2v10a2 2 0 002 2h8a2 2 0 002-2V6a1 1 0 100-2h-3.382l-.724-1.447A1 1 0 0011 2H9zM7 8a1 1 0 012 0v6a1 1 0 11-2 0V8zm5-1a1 1 0 00-1 1v6a1 1 0 102 0V8a1 1 0 00-1-1z"
                          clipRule="evenodd"
                        />
                      </svg>
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </main>

      <ConfirmationModal
        isOpen={deleteModalOpen}
        onClose={() => {
          setDeleteModalOpen(false);
          setProjectToDelete(null);
        }}
        onConfirm={handleDeleteConfirm}
        title="Delete Project"
        message={`Are you sure you want to delete "${projectToDelete?.name}"? This action cannot be undone.`}
        confirmText={isDeleting ? "Deleting..." : "Delete"}
        cancelText="Cancel"
      />
    </div>
  );
}
