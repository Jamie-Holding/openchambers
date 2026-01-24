import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env file from project root
env_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(env_path)

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://localhost:5432/hansard"
)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
)
AGENT_MODEL_NAME = os.getenv("AGENT_MODEL_NAME", "gpt-4o-mini")
AGENT_DEBUG = os.getenv("AGENT_DEBUG", "false").lower() == "true"
