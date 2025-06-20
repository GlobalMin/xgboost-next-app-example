# XGBoost Simple GUI App

A very simple example of turning a Jupyter notebook code heavy work flow for XGBoost into a GUI based app. This all runs completely locally and for free. It's written with you as the single user in mind. There already exist plenty of production ready ML enterprise systems out there, this is to show how quickly ideas can turn into an MVP working version. Learning the basics of frameworks you come across like NextJS or FastAPI for my case will guard against LLM agents making bad choices without you seeing.

3-4 years ago this might have taken me 6 months and a lot of almost giving up moments versus today in 2025 I made this whole repo and project in 2 days with Claude Code mostly. 

## Project Purpose

This application serves as a learning platform for:

- **Rapid Prototyping**: Practice turning ML concepts into functional applications quickly
- **MVP Development**: Build the most basic version that demonstrates core functionality
- **Ownership Mindset**: Create tools that serve your specific needs rather than relying solely on external platforms

The goal isn't to build production-grade ML infrastructure (that's not feasible for individual projects), but to strengthen the muscle of transforming ideas into working prototypes. In today's AI-powered development landscape, this skill becomes increasingly valuable.

## 🏗️ Architecture & Key Concepts

### **Dual-Service Architecture**

- **Frontend**: Next.js React application (<http://localhost:3000>)
- **Backend**: FastAPI Python API (<http://localhost:8000>)
- **Database**: MongoDB for persistent data storage

### **Concepts Demonstrated**

✅ **File Upload & Storage**: CSV data handling and model artifact persistence  
✅ **Database Persistence**: MongoDB integration for data that survives browser sessions  
✅ **Real-time Progress**: Live training status updates and progress tracking  
✅ **Data Visualization**: Interactive charts for model performance analysis  
✅ **API Communication**: Frontend-backend data exchange patterns  

### **Intentionally Excluded Concepts**

❌ **Authentication**: Single-user local application doesn't require user management  
❌ **Worker Queues**: Long-running tasks handled synchronously for simplicity  
❌ **Production Scaling**: Focus on learning core concepts, not enterprise deployment  

## Getting Started

### Prerequisites

1. **Node.js** (v18 or higher) and npm
2. **Python 3.12+** with [uv package manager](https://docs.astral.sh/uv/)
3. **MongoDB** - Install [MongoDB Compass](https://www.mongodb.com/products/compass) for free local database management

### Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd xgboost_simple_gui_app
   ```

2. **Set up the Python backend**

   ```bash
   # Install Python dependencies
   uv sync
   
   # Activate virtual environment
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Set up the Node.js frontend**

   ```bash
   # Install frontend dependencies
   npm install
   ```

4. **Configure MongoDB**
   - Start MongoDB service locally (via MongoDB Compass or command line)
   - Copy `.env.sample` to `.env.local`
   - Update MongoDB connection settings if needed

5. **Environment Configuration**

   ```bash
   cp .env.sample .env.local
   # Edit .env.local with your MongoDB settings
   ```

### Running the Application

You need to run both services in separate terminals:

**Terminal 1 - Backend API:**

```bash
# Option 1: Use the provided script
./start-backend.sh

# Option 2: Manual startup
source .venv/bin/activate
cd python_api
uvicorn main:app --reload --port 8000
```

**Terminal 2 - Frontend:**

```bash
npm run dev
```

The application will be available at:

- Frontend: <http://localhost:3000>
- Backend API: <http://localhost:8000>
- API Documentation: <http://localhost:8000/docs>

## 📊 Usage Guide

### 1. **Upload CSV Data**

- Click "Upload CSV" on the home page
- Select a CSV file with your dataset
- Preview the data structure and columns

### 2. **Configure Model Training**

- Select target column (what you want to predict)
- Choose feature columns (input variables)
- Adjust training parameters:
  - Test/train split ratio
  - Cross-validation folds
  - Parameter tuning options
  - Early stopping rounds

### 3. **Monitor Training**

- Real-time progress updates during model training
- Training logs and status information
- Automatic completion detection

### 4. **Analyze Results**

- Model performance metrics (accuracy, precision, recall, F1-score)
- Feature importance rankings
- Lift chart visualization
- Confusion matrix analysis

### 5. **Model Management**

- Browse recently trained models
- Look up models by ID
- Compare different model configurations

## 📁 Project Structure

```
xgboost_simple_gui_app/
├── app/                          # Next.js app directory
│   ├── page.tsx                  # Main application page
│   ├── layout.tsx                # App layout and styling
│   └── globals.css               # Global styles
├── components/                   # React components
│   ├── csv-upload.tsx            # File upload interface
│   ├── data-preview.tsx          # Dataset preview and config
│   ├── training-progress.tsx     # Real-time training status
│   ├── model-results.tsx         # Results display
│   ├── lift-chart.tsx            # Performance visualization
│   ├── model-lookup.tsx          # Model search functionality
│   └── recent-models.tsx         # Recent models list
├── python_api/                   # FastAPI backend
│   ├── main.py                   # API server entry point
│   ├── routes.py                 # API endpoints
│   ├── models.py                 # Database models
│   ├── xgb_modeling.py           # XGBoost training logic
│   ├── modeling_utils.py         # ML utility functions
│   ├── mongo_utils.py            # MongoDB operations
│   ├── config.py                 # Configuration settings
│   ├── uploads/                  # CSV file storage
│   └── models/                   # Trained model artifacts
├── package.json                  # Frontend dependencies
├── pyproject.toml                # Python dependencies
├── start-backend.sh              # Backend startup script
└── .env.local                    # Environment configuration
```

## 🛠️ Development Notes

### **MVP Philosophy**

This project embodies MVP (Minimum Viable Product) principles:

- **Core functionality first**: Focus on essential features that demonstrate the concept
- **Iterative improvement**: Build basic version, then enhance based on usage
- **Learning-oriented**: Prioritize understanding over perfection

### **Technology Choices**

- **Next.js**: Rapid frontend development with React
- **FastAPI**: Fast Python API development with automatic documentation
- **MongoDB**: Flexible document storage for ML metadata
- **XGBoost**: Industry-standard gradient boosting for reliable ML results

### **Scaling Considerations**

For production use, consider adding:

- User authentication and authorization
- Background job processing for long-running training
- Model versioning and experiment tracking
- Automated model deployment pipelines
- Monitoring and logging infrastructure

