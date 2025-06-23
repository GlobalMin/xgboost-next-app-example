# XGBoost training GUI Next.js + FastAPI

## Background

For a number of years now, xgboost has stayed consistently at the top of ML alogorithms for binary classification use cases. When working with new datasets and projects, I'll usually end up running a quick benchmark using xgboost to get a rough floor of the model performance and what are the top features. I finally just went for it and built a protoype GUI version of this exercise for my needs using Claude Code.

![2025-06-22 18 02 59](https://github.com/user-attachments/assets/47bf66b9-dd28-4bf9-a95c-a39a05239bc9)

This app runs locally and focuses specifically on the workflow I found myself repeating: load data, select features based on signal strength, train with automated parameter tuning, and visualize performance. It's not a production ML platform - there's no authentication, deployment pipelines, or cloud infrastructure. Just a focused tool for getting quick model benchmarks and understanding feature importance.

### Using Claude Code as LLM agent

There are now a number of popular Agentic Coding VSCode forks like Curso and Windsurf as ewll as extensions that work wtihin VSCode and other IDE's like Cline. Claude Code is a terminal based interface that seems to be able to handle more challenging/longeer coding tasks than the ones mentioned above. Any of them are probably fine to use at this point, but I chose to test out Claude Code for this app.

The biggest takeaways on how to best work with an LLM Agent are:

1) Try to start the project with some kind of opinions on tech stack and code design. LLMs don't do well with unconstrained requests so give some guidance while you build up patterns in the codebase.
2) Leverage Claude's task plannning functionality or if using another agent start with generating a detailed plan that you output to a markdown file with check boxes that the LLM fills in as it progresses. Written plans are able to exist outside of 1 conversation's context window.
3) Commit and push changes very frequently. You lose nothing by doing this a little too much and if you don't make a habit of it you will mess up the code base somehow that forces you to revert back to an older commit and will have to redo anything in between. More commits means more places to revert back to if needed.

Ok, thank you for listening to my TED Talk. 👏

## Requirements

- Node.js v18+
- Python 3.12+
- MongoDB (local instance)
- [uv](https://docs.astral.sh/uv/) Python package manager

## Setup

Clone and install:

```bash
git clone <repository-url>
cd xgboost_simple_gui_app
uv sync
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
npm install
```

Configure environment:

```bash
cp .env.sample .env.local
# Edit .env.local if your MongoDB setup differs
```

## Running

Start both services:

```bash
# Terminal 1: Backend
./start-backend.sh

# Terminal 2: Frontend
npm run dev
```

Open <http://localhost:3000>

## Workflow

The application streamlines the typical XGBoost training process:

1. Upload your CSV dataset via drag-and-drop
2. Select the target column for binary classification
3. Use the f-score calculation to identify features with strong signal
4. Train the model with automatic hyperparameter tuning
5. Review performance metrics, lift charts, and feature importance
6. Export the trained model as Python code for production use

The entire process typically completes in under 5 minutes, significantly faster than writing custom training scripts for each new dataset.

## Final Thoughts

This tool demonstrates both the potential and limitations of AI-assisted development. While the code may not exemplify every software engineering best practice, it successfully solves a real problem and provides a useful workflow for XGBoost model development.

If you find this approach useful or have suggestions for improvements, I'd welcome your feedback. The repository is open for forking and adaptation to your own workflows.
