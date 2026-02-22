"""Shared instances for agent nodes."""

from langchain_openai import ChatOpenAI

from backend.config.settings import AGENT_MODEL_NAME, EMBEDDING_MODEL_NAME
from backend.src.chatbot.tools import HansardRetrievalTool

llm = ChatOpenAI(model=AGENT_MODEL_NAME, temperature=0.0, top_p=1.0)
hansard_tool = HansardRetrievalTool(
    model_name=EMBEDDING_MODEL_NAME, top_k=10, min_similarity=0.1
)
