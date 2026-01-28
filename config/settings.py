import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
LOG_DIR = os.path.join(BASE_DIR, "logs")

# Model paths
MODEL_PATH = os.path.join(MODEL_DIR, "fake_review_detector.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "feature_scaler.pkl")

# API configuration
API_HOST = "0.0.0.0"
API_PORT = 5000
API_DEBUG = True
