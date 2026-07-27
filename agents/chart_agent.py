# agents/chart_agent.py
# ------------------------
# WHY: Picking the "right" chart for a dataset is a skill - a bar chart makes
#      sense for categories, a scatter plot for two numeric columns, etc.
#      This agent demonstrates "tool calling": the LLM DECIDES which chart
#      and columns to use, then a separate tool function actually DRAWS it.
# WHAT: The fourth node in our LangGraph workflow. Asks Gemini to recommend
#       a chart type + axes, then calls tools/chart_tools.py to render it.
# HOW: We ask the LLM to reply in a simple, strict JSON format, parse that
#      JSON, and pass the result into our chart-drawing tool.

import json
from graph.state import AgentState
from tools.chart_tools import create_chart, SUPPORTED_CHART_TYPES
from utils.llm import get_llm
from llmops.token_manager import extract_token_usage


CHART_RECOMMENDATION_PROMPT_TEMPLATE = """
You are a data visualization expert. A dataset has these columns:
Numeric columns: {numeric_columns}
Categorical columns: {categorical_columns}

Supported chart types: {supported_chart_types}

Recommend the SINGLE best chart to visualize this dataset. Reply with ONLY
valid JSON (no extra text, no markdown fences) in exactly this shape:
{{
  "chart_type": "one of the supported chart types",
  "x_axis": "a column name from the dataset",
  "y_axis": "a column name from the dataset, or null if not needed",
  "reason": "one short sentence explaining why this chart is a good choice"
}}
"""


def run_chart_agent(state: AgentState) -> AgentState:
    """
    WHY: This is the LangGraph node that demonstrates "tool calling" - the
         LLM makes a DECISION (which chart/columns), and a plain Python tool
         function EXECUTES that decision (draws the chart).
    WHAT: Reads the DataFrame from state, asks Gemini to recommend a chart,
          then generates the actual Plotly figure and stores it in state.
    HOW: 1) Ask the LLM for a JSON recommendation.
         2) Parse the JSON safely (with a simple fallback if parsing fails).
         3) Call tools/chart_tools.create_chart() to render the figure.
    """
    df = state["uploaded_dataframe"]
    profiling_result = state.get("profiling_result", {})

    numeric_columns = profiling_result.get("numeric_columns", df.select_dtypes(include="number").columns.tolist())
    categorical_columns = profiling_result.get("categorical_columns", df.select_dtypes(include="object").columns.tolist())

    # If the user already picked a chart type manually (from the Visualization
    # page dropdown), we skip asking the LLM and just use their choice.
    if state.get("chart_type") and state.get("chart_x_axis"):
        chart_type = state["chart_type"]
        x_axis = state["chart_x_axis"]
        y_axis = state.get("chart_y_axis")
        reason = "Selected manually by the user."
    else:
        # Ask the LLM to recommend the best chart type and columns.
        prompt = CHART_RECOMMENDATION_PROMPT_TEMPLATE.format(
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            supported_chart_types=SUPPORTED_CHART_TYPES,
        )
        llm = get_llm()
        response = llm.invoke(prompt)

        # LLMOps: record how many tokens this call used (see llmops/token_manager.py).
        state["token_usage"] = extract_token_usage(response)

        # The LLM should return JSON text. We parse it carefully, and fall
        # back to a safe default chart if anything goes wrong (e.g. the LLM
        # added extra text around the JSON).
        try:
            cleaned_text = response.content.strip().strip("`").replace("json\n", "")
            recommendation = json.loads(cleaned_text)
            chart_type = recommendation.get("chart_type", "Bar")
            x_axis = recommendation.get("x_axis", categorical_columns[0] if categorical_columns else df.columns[0])
            y_axis = recommendation.get("y_axis", numeric_columns[0] if numeric_columns else None)
            reason = recommendation.get("reason", "Recommended by the AI Chart Agent.")
        except (json.JSONDecodeError, IndexError):
            # Safe fallback so the app never crashes if the LLM output is malformed.
            chart_type = "Bar"
            x_axis = categorical_columns[0] if categorical_columns else df.columns[0]
            y_axis = numeric_columns[0] if numeric_columns else None
            reason = "Fallback chart chosen because the AI response could not be parsed."

    # Use our chart-drawing tool to build the actual Plotly figure.
    figure = create_chart(df, chart_type, x_axis, y_axis)

    # Save everything back into the shared state.
    state["chart_type"] = chart_type
    state["chart_x_axis"] = x_axis
    state["chart_y_axis"] = y_axis
    state["chart_reason"] = reason
    state["generated_chart"] = figure

    # Log this step for the execution trace.
    messages = state.get("messages", [])
    messages.append({"role": "agent", "content": f"Chart Agent: recommended a {chart_type} chart. Reason: {reason}"})
    state["messages"] = messages

    return state
