# llmops/model_serving.py
# --------------------------
# WHY: "Model Serving" means wrapping your AI logic in small, reusable
#      functions that DON'T care who's calling them - a Streamlit button, a
#      REST API, a scheduled job, a test script. If the logic only lives
#      inside a Streamlit button's on_click code, nothing else can reuse it.
# WHAT: Three "service" functions that wrap our existing agents around a
#       bundled sample dataset (chosen by name, e.g. "employees" or "sales").
#       These are the SAME functions used by:
#         1. llmops/api_server.py (a real HTTP API)
#         2. app.py's LLMOps demo page (calls them directly, in-process)
#       This is the essence of model serving: one implementation, many callers.
# HOW: Plain functions - load the CSV, call our existing tools/agents, return
#      a plain (JSON-safe) dictionary.

import os
import pandas as pd

from tools.dataframe_tools import build_full_profile
from agents.insight_agent import run_insight_agent
from agents.rag_agent import run_rag_agent

# Where the bundled sample datasets live, relative to the project root.
DATA_FOLDER = "data"

# The datasets students can pick from in the demo - keeps the API's input
# surface small and safe (no arbitrary file paths from a caller).
AVAILABLE_DATASETS = ["employees", "sales"]


def _load_dataset(dataset_name: str) -> pd.DataFrame:
    """
    WHY: Every service function below needs the same "load a bundled CSV by
         name" step, so we do it once here.
    WHAT: Reads data/<dataset_name>.csv into a DataFrame.
    HOW: Validates the name against AVAILABLE_DATASETS first, so a caller
         can't ask us to read an arbitrary file off disk.
    """
    if dataset_name not in AVAILABLE_DATASETS:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Choose one of: {AVAILABLE_DATASETS}")

    csv_path = os.path.join(DATA_FOLDER, f"{dataset_name}.csv")
    return pd.read_csv(csv_path)


def serve_profile(dataset_name: str) -> dict:
    """
    WHY: "Serves" the Profiler Agent's output as a plain dictionary - no
         Streamlit, no LangGraph state, just data in -> data out.
    WHAT: Returns the full profiling dictionary for the requested dataset.
    """
    df = _load_dataset(dataset_name)
    return build_full_profile(df)


def serve_insights(dataset_name: str) -> dict:
    """
    WHY: "Serves" the Insight Agent (which calls the LLM) as a plain function.
    WHAT: Returns {"insights": "...", "token_usage": {...}}.
    """
    df = _load_dataset(dataset_name)
    profiling_result = build_full_profile(df)

    state = {"profiling_result": profiling_result}
    state = run_insight_agent(state)

    return {
        "dataset": dataset_name,
        "insights": state["ai_insights"],
        "token_usage": state.get("token_usage"),
    }


def serve_ask(dataset_name: str, question: str) -> dict:
    """
    WHY: "Serves" the RAG Agent as a plain function - this is what a chatbot
         widget on a website, or a Slack bot, would call behind the scenes.
    WHAT: Returns {"question": ..., "answer": ..., "retrieved_documents": [...]}.
    """
    df = _load_dataset(dataset_name)

    state = {"uploaded_dataframe": df, "user_question": question}
    state = run_rag_agent(state)

    return {
        "dataset": dataset_name,
        "question": question,
        "answer": state["rag_answer"],
        "retrieved_documents": state["retrieved_documents"],
        "token_usage": state.get("token_usage"),
    }
