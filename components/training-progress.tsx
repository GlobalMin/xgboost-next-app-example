'use client';

import React, { useEffect, useState } from 'react';
import { Loader2, CheckCircle, Circle } from 'lucide-react';

interface TrainingProgressProps {
  modelId: number | null;
}

interface ProgressLog {
  message: string;
  timestamp: string;
}

interface TrainingStep {
  name: string;
  description: string;
  completed: boolean;
  active: boolean;
}

export function TrainingProgress({ modelId }: TrainingProgressProps) {
  const [logs, setLogs] = useState<ProgressLog[]>([]);
  const [status, setStatus] = useState<string>('training');
  const [steps, setSteps] = useState<TrainingStep[]>([
    { name: 'Data Preparation', description: 'Loading and preprocessing data', completed: false, active: true },
    { name: 'Parameter Tuning', description: 'Finding optimal parameters via grid search', completed: false, active: false },
    { name: 'Model Training', description: 'Training final model with best parameters', completed: false, active: false },
    { name: 'Evaluation', description: 'Calculating performance metrics', completed: false, active: false },
  ]);

  useEffect(() => {
    if (!modelId) return;

    const eventSource = new EventSource(`http://localhost:8000/api/models/${modelId}/progress`);
    
    eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.error) {
        console.error('Progress error:', data.error);
        eventSource.close();
        return;
      }
      
      setStatus(data.status);
      setLogs(data.logs || []);
      
      // Update steps based on logs
      const logMessages = (data.logs || []).map(log => log.message).join(' ');
      
      setSteps(prevSteps => {
        const newSteps = [...prevSteps];
        
        // Data Preparation
        if (logMessages.includes('Dataset loaded') || logMessages.includes('Train/Test split')) {
          newSteps[0].completed = true;
          newSteps[0].active = false;
          newSteps[1].active = true;
        }
        
        // Parameter Tuning
        if (logMessages.includes('Starting grid search') || logMessages.includes('Grid search completed')) {
          if (logMessages.includes('Grid search completed')) {
            newSteps[1].completed = true;
            newSteps[1].active = false;
            newSteps[2].active = true;
          }
        }
        
        // Model Training
        if (logMessages.includes('Training final model')) {
          newSteps[2].active = true;
          if (logMessages.includes('Training completed!')) {
            newSteps[2].completed = true;
            newSteps[2].active = false;
            newSteps[3].active = true;
          }
        }
        
        // Evaluation
        if (logMessages.includes('Training completed!')) {
          newSteps[3].completed = true;
          newSteps[3].active = false;
        }
        
        return newSteps;
      });
      
      if (data.status === 'completed' || data.status === 'failed') {
        eventSource.close();
      }
    };

    eventSource.onerror = (error) => {
      console.error('EventSource error:', error);
      eventSource.close();
    };

    return () => {
      eventSource.close();
    };
  }, [modelId]);

  return (
    <div className="w-full max-w-6xl mx-auto p-6 space-y-6">
      <div className="bg-white rounded-lg shadow p-6">
        <div className="flex items-center space-x-3 mb-6">
          <Loader2 className="h-6 w-6 animate-spin text-blue-600" />
          <h3 className="text-xl font-semibold">Training Your Model</h3>
        </div>
        
        {/* Progress Steps */}
        <div className="space-y-4 mb-6">
          {steps.map((step, index) => (
            <div key={index} className="flex items-start space-x-3">
              <div className="mt-0.5">
                {step.completed ? (
                  <CheckCircle className="h-5 w-5 text-green-600" />
                ) : step.active ? (
                  <Loader2 className="h-5 w-5 text-blue-600 animate-spin" />
                ) : (
                  <Circle className="h-5 w-5 text-gray-300" />
                )}
              </div>
              <div className="flex-1">
                <p className={`font-medium ${
                  step.completed ? 'text-green-700' : 
                  step.active ? 'text-blue-700' : 
                  'text-gray-500'
                }`}>
                  {step.name}
                </p>
                <p className="text-sm text-gray-600">{step.description}</p>
              </div>
            </div>
          ))}
        </div>
        
        {/* Logs Section */}
        <div className="border-t pt-4">
          <h4 className="text-sm font-medium text-gray-700 mb-2">Training Logs</h4>
          <div className="bg-gray-50 rounded-lg p-3 max-h-48 overflow-y-auto">
            {logs.length === 0 ? (
              <p className="text-gray-500 text-sm">Initializing training process...</p>
            ) : (
              <div className="space-y-1">
                {logs.slice().reverse().slice(0, 10).map((log, index) => (
                  <div key={index} className="text-xs">
                    <span className="text-gray-500 font-mono">
                      {new Date(log.timestamp).toLocaleTimeString()}
                    </span>
                    <span className="ml-2 text-gray-700">{log.message}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}