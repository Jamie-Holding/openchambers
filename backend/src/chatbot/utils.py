"""Shared instances for agent nodes."""

from langchain_openai import ChatOpenAI

from config.settings import EMBEDDING_MODEL_NAME, FAST_LLM_MODEL_NAME, LLM_MODEL_NAME
from src.chatbot.tools import HansardRetrievalTool

llm = ChatOpenAI(model=LLM_MODEL_NAME, temperature=0.0, top_p=1.0)
fast_llm = ChatOpenAI(model=FAST_LLM_MODEL_NAME, temperature=0.0, top_p=1.0)
hansard_tool = HansardRetrievalTool(
    model_name=EMBEDDING_MODEL_NAME, top_k=10, min_similarity=0.1
)
