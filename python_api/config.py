import os

UPLOAD_DIR = "uploads"
MODEL_DIR = "models"


# Create directories if they don't exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
