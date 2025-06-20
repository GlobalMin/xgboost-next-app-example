# XGBoost Simple GUI App

A Next.js application for training XGBoost models with a simple drag-and-drop interface.

## Features

- CSV file upload with drag-and-drop
- Data preview and column selection
- XGBoost model training with configurable parameters
- Real-time training progress monitoring
- Model evaluation metrics (AUC, Lift Chart)
- Feature importance visualization
- Parameter tuning with grid search
- Cross-validation with stratified K-folds
- Early stopping to prevent overfitting
- Model results lookup by ID
- Recent models list with quick access

## Setup

1. Install Node.js dependencies:
```bash
npm install
```

2. Set up Python environment:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running the Application

1. Start the FastAPI backend:
```bash
./start-backend.sh
```

2. In a new terminal, start the Next.js frontend:
```bash
npm run dev
```

3. Open http://localhost:3000 in your browser

## Usage

### Training a New Model
1. Upload a CSV file using the drag-and-drop interface
2. Select the target column for prediction
3. Choose feature columns
4. Configure model parameters (optional):
   - Test set size
   - Number of CV folds
   - Enable parameter tuning
   - Early stopping rounds
5. Click "Train Model" to start training
6. Monitor real-time training progress
7. View results including AUC score, feature importance, and lift chart

### Viewing Previous Models
1. Enter a model ID in the "View Previous Model Results" input
2. Click "View Results" to retrieve model performance metrics
3. Or click on any model in the "Recent Models" list

## Tech Stack

- Frontend: Next.js, React, TypeScript, Tailwind CSS
- Backend: FastAPI, Python
- ML: XGBoost (native API), scikit-learn
- Database: SQLite (MongoDB Atlas migration planned - see MONGODB_MIGRATION_PLAN.md)
- Charts: Recharts
- Real-time updates: Server-Sent Events (SSE)