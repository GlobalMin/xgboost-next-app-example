#!/bin/bash

echo "Starting FastAPI backend..."
source .venv/bin/activate
echo "Activated virtual environment."

cd python_api
uvicorn main:app --reload --port 8000