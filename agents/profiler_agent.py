# agents/profiler_agent.py
# --------------------------
# WHY: Before we can generate AI insights, we need to understand the RAW
#      shape of the data - how many rows, missing values, column types, etc.
#      This is traditionally called "data profiling".
# WHAT: The first node in our LangGraph workflow. It does NOT call the LLM -
#       it's a good example of an "agent" that is really just a smart
#       pandas function wrapped in the LangGraph node interface.
# HOW: Calls our reusable dataframe_tools.build_full_profile() function and
#      stores the structured result into the shared graph state.

from graph.state import AgentState
from tools.dataframe_tools import build_full_profile


def run_profiler_agent(state: AgentState) -> AgentState:
    """
    WHY: This function is registered as a LangGraph "node". LangGraph will
         call it automatically as part of the workflow, passing in the
         current state and expecting an updated state back.
    WHAT: Reads the uploaded DataFrame from state, profiles it, and writes
          the structured profiling_result back into state.
    HOW: Delegates the actual pandas work to tools/dataframe_tools.py to
         keep this agent file short and focused on the "agent" responsibility
         (reading/writing state), not the low-level data logic.
    """
    df = state["uploaded_dataframe"]

    # Do the actual profiling work using our reusable tool function.
    profiling_result = build_full_profile(df)

    # Write the result back into the shared state so later agents (and the
    # Streamlit UI) can use it.
    state["profiling_result"] = profiling_result

    # Log this step in the shared "messages" list - useful for the
    # bonus "Agent Execution Trace" feature in the UI.
    messages = state.get("messages", [])
    messages.append({"role": "agent", "content": "Profiler Agent: dataset profiled successfully."})
    state["messages"] = messages

    return state
