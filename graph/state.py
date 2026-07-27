# graph/state.py
# ----------------
# WHY: LangGraph works by passing a single "state" dictionary between agents
#      (nodes). Each agent reads what it needs from the state and writes its
#      results back into it. Defining the shape of that state in one place
#      makes it easy to see exactly what data flows through our workflow.
# WHAT: Defines AgentState - a TypedDict describing every field our graph uses.
# HOW: We use Python's typing.TypedDict, which is just a regular dictionary
#      with type hints. It does NOT enforce types at runtime - it just helps
#      humans (and editors) understand what's inside the state.

from typing import TypedDict, Optional, List, Dict, Any
import pandas as pd


class AgentState(TypedDict, total=False):
    """
    This is the "memory" that is shared between every agent in our LangGraph
    workflow. Think of it as a shared notebook that gets passed from agent
    to agent - each agent reads a few pages and writes a few new pages.

    total=False means none of these keys are strictly required, which keeps
    things flexible and beginner-friendly (we don't have to fill in every
    field every single time).
    """

    # --- Input data ---
    uploaded_dataframe: pd.DataFrame        # The raw CSV loaded into a pandas DataFrame
    dataset_name: str                       # A friendly name, e.g. "employees.csv"

    # --- Written by the Profiler Agent ---
    profiling_result: Dict[str, Any]        # Shape, missing values, dtypes, summary stats, etc.

    # --- Written by the Insight Agent ---
    ai_insights: str                        # LLM-generated business insights (markdown text)

    # --- Used/written by the RAG Agent ---
    user_question: Optional[str]            # The natural-language question the user typed
    retrieved_documents: List[str]          # Top-k text chunks retrieved from FAISS
    rag_answer: Optional[str]               # The LLM's answer to user_question

    # --- Used/written by the Chart Agent ---
    chart_requested: bool                   # True if the user wants a chart generated
    chart_type: Optional[str]               # e.g. "bar", "line", "pie", "scatter"
    chart_x_axis: Optional[str]             # Column name suggested for the X axis
    chart_y_axis: Optional[str]             # Column name suggested for the Y axis
    chart_reason: Optional[str]             # Why the LLM chose this chart
    generated_chart: Optional[Any]          # The actual Plotly figure object

    # --- Written by the Report Agent ---
    report: Optional[str]                   # Final markdown report text

    # --- Shared conversation log (LangChain "messages" concept) ---
    messages: List[Dict[str, str]]          # A running log of what each agent said/did

    # --- Bonus: execution trace for the UI ("which agent ran, how long") ---
    agent_trace: List[Dict[str, Any]]

    # --- LLMOps: token usage from the most recent LLM call (see llmops/token_manager.py) ---
    token_usage: Dict[str, int]
