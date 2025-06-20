from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from dotenv import load_dotenv

from routes import router


# Load environment variables
load_dotenv(override=True)


app = FastAPI(title="XGBoost Model Training API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the router with all routes
app.include_router(router)


@app.get("/")
async def root():
    return {"message": "XGBoost Model Training API", "version": "2.0"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
