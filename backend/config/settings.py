import os

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://localhost:5432/hansard"
)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
)
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")
FAST_LLM_MODEL_NAME = os.getenv("FAST_LLM_MODEL_NAME", "gpt-4.1-nano")
AGENT_DEBUG = os.getenv("AGENT_DEBUG", "false").lower() == "true"
