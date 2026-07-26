# agents/report_agent.py
# -------------------------
# WHY: After all other agents have done their work (profiling, insights,
#      Q&A, charts), someone needs to collect everything into one shareable
#      document. That's the Report Agent's job - it's the FINAL node in our
#      LangGraph workflow.
# WHAT: The fifth node in our LangGraph workflow. Reads everything the other
#       agents produced from state and writes a single markdown report.
# HOW: String formatting only - no LLM call needed here, since we already
#      have all the AI-generated text from earlier agents.

from graph.state import AgentState
from utils.helpers import get_timestamp, ensure_folder_exists
import os


def run_report_agent(state: AgentState) -> AgentState:
    """
    WHY: This is the LangGraph node that assembles the final deliverable -
         a markdown report a student could hand to a manager or teacher.
    WHAT: Reads profiling_result, ai_insights, chart info, and rag_answer
          from state, and writes a formatted markdown report into state["report"].
    HOW: Builds the report as a plain Python string using f-strings, then
         saves it to the reports/ folder.
    """
    dataset_name = state.get("dataset_name", "Uploaded Dataset")
    profiling_result = state.get("profiling_result", {})
    ai_insights = state.get("ai_insights", "No AI insights were generated.")
    chart_type = state.get("chart_type")
    chart_reason = state.get("chart_reason")
    user_question = state.get("user_question")
    rag_answer = state.get("rag_answer")
    timestamp = get_timestamp()

    # Build the report section by section. This is intentionally simple
    # string concatenation so students can see exactly how the report forms.
    report_lines = [
        f"# AI Data Scientist Report",
        f"**Dataset:** {dataset_name}",
        f"**Generated on:** {timestamp}",
        "",
        "## 1. Dataset Summary",
        f"- Rows: {profiling_result.get('shape', {}).get('rows', 'N/A')}",
        f"- Columns: {profiling_result.get('shape', {}).get('columns', 'N/A')}",
        f"- Duplicate rows: {profiling_result.get('duplicate_rows', 'N/A')}",
        f"- Numeric columns: {', '.join(profiling_result.get('numeric_columns', []))}",
        f"- Categorical columns: {', '.join(profiling_result.get('categorical_columns', []))}",
        "",
        "## 2. AI-Generated Insights",
        ai_insights,
        "",
    ]

    # Only add the Q&A section if the user actually asked something.
    if user_question and rag_answer:
        report_lines += [
            "## 3. Ask Your Dataset (RAG) Q&A",
            f"**Question:** {user_question}",
            f"**Answer:** {rag_answer}",
            "",
        ]

    # Only add the chart section if a chart was generated.
    if chart_type:
        report_lines += [
            "## 4. Visualization",
            f"- Recommended chart: **{chart_type}**",
            f"- Reason: {chart_reason}",
            "",
        ]

    report_lines += [
        "## 5. Recommendations & Next Steps",
        "See the AI-Generated Insights section above for detailed recommendations.",
        "",
        "---",
        "*Report generated automatically by the AI Data Scientist LangGraph workflow.*",
    ]

    report_text = "\n".join(report_lines)
    state["report"] = report_text

    # Save the report to disk so it can be downloaded later.
    ensure_folder_exists("reports")
    file_name = f"report_{timestamp.replace(':', '-').replace(' ', '_')}.md"
    file_path = os.path.join("reports", file_name)
    with open(file_path, "w", encoding="utf-8") as report_file:
        report_file.write(report_text)

    # Log this step for the execution trace.
    messages = state.get("messages", [])
    messages.append({"role": "agent", "content": f"Report Agent: report saved to {file_path}."})
    state["messages"] = messages

    return state
