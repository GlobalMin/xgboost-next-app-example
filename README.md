# Simple XGBoost GUI - An Example of LLM Agent Development

## Overview

A streamlined web application for XGBoost binary classification model training, built with Next.js and FastAPI. This tool automates the workflow of loading datasets, training models, and generating performance insights including lift charts and feature importance analysis.

![enter image description here](https://github-production-user-asset-6210df.s3.amazonaws.com/29800959/457685182-836fbe61-c253-475c-8171-1bb847f97680.gif?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20250622%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20250622T220832Z&X-Amz-Expires=300&X-Amz-Signature=4044092b16a68019773170de9d118dcb5e7174371bec93d5c2a0c7f89b4b7135&X-Amz-SignedHeaders=host)

**Purpose**: Replace manual XGBoost model training workflows with a simple GUI that covers 90%+ of common binary classification tasks.

**Scope**: This is a personal productivity tool designed for local use, not a general-purpose ML platform. It's optimized for specific workflows and may not achieve acceptable performance for all use cases.

## What This Is (and Isn't)

### What it is:
- A working example of rapid prototyping with LLM agents
- A practical tool for quick binary classification benchmarking
- A localhost-only application for single users

### What it isn't:
- A production-ready ML platform
- A best practices template for Next.js/FastAPI
- A general-purpose solution for all ML workflows

No authentication, cloud storage, or deployment pipelines are included by design. This keeps the tool simple and focused on its core purpose.

## Built with Claude Code - Insights on LLM Agent Development

This repository demonstrates the dramatic acceleration possible with LLM agents like Claude Code. The key to success is bringing the right human expertise to guide the AI effectively.

### Success Factors:

1. **Domain Knowledge**: Understanding the technical requirements and ideal architecture helps steer the AI toward good solutions
2. **Clear Milestones**: Breaking the project into specific, small tasks minimizes the risk of the agent going off track
3. **Stack Familiarity**: Having experience with FastAPI and knowledge of Next.js capabilities allowed for informed decisions
4. **Active Review**: Regularly reviewing generated code catches issues early

### Development Strategy:

- **Commit frequently** - Better to over-commit than lose good work when reverting bad suggestions
- **Set boundaries** - The more specific your requirements, the better the outcomes
- **Stay engaged** - Pure "vibe coding" without review will accumulate technical debt quickly
- **Know when to intervene** - Sometimes you need to manually correct course before the agent compounds mistakes

The combination of human judgment and AI speed creates a powerful development workflow. You bring the vision and quality control; the LLM handles the implementation details.

## Technical Requirements

- Node.js v18+
- Python 3.12+
- MongoDB (local instance)
- [uv](https://docs.astral.sh/uv/) Python package manager

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd xgboost_simple_gui_app
```

2. Set up Python environment:
```bash
uv sync
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install frontend dependencies:
```bash
npm install
```

4. Configure environment:
```bash
cp .env.sample .env.local
# Edit .env.local to match your MongoDB setup if needed
```

## Running the Application

Start both services:

```bash
# Terminal 1: Backend API
./start-backend.sh

# Terminal 2: Frontend
npm run dev
```

Navigate to http://localhost:3000

## Workflow

1. **Upload Data**: Drag and drop your CSV file
2. **Configure Model**:
   - Select target column
   - Click "Check Features for Signal" to calculate f-scores
   - Select features based on signal strength
3. **Train**: Click "Train Model" for automatic parameter tuning
4. **Review Results**:
   - Model metrics (AUC, accuracy, precision, recall)
   - Lift chart showing model performance vs. random selection
   - Feature importance rankings
5. **Export**: Generate Python code for production deployment

## Final Thoughts

This project exists because LLM agents have made it feasible to build custom tools in days rather than months. It's not perfect code, but it's working code that solves a real problem. For developers with limited time but specific needs, this new paradigm is game-changing.

Feel free to clone and adapt for your own workflows. If you find it useful or have feedback on the specific implementation, I'd be interested to hear your thoughts.