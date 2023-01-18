import os
import torch
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path.cwd() / ".env")

GDB_URL = os.getenv("GDB_URL")
GBD_PORT = os.getenv("GBD_PORT")
GDB_USER = os.getenv("GDB_USER")
GDB_PASSWORD = os.getenv("GDB_PASSWORD")
MLFLOW_URL_PREFIX = os.getenv("MLFLOW_URL_PREFIX")
MLFLOW_USER = os.getenv("MLFLOW_USER")
MLFLOW_PASSWORD = os.getenv("MLFLOW_PASSWORD")
MLFLOW_URL = os.getenv("MLFLOW_URL")
MLFLOW_PORT = os.getenv("MLFLOW_PORT")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
