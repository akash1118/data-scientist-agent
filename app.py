# app.py
# -------
# WHY: This is the ONE file that runs the entire application. We deliberately
#      avoid extra web frameworks (like FastAPI) - everything, including
#      navigation between "pages", lives inside this single Streamlit script.
# WHAT: A 7-page beginner-friendly "AI Data Scientist" app that lets a student
#       upload a CSV, profile it, get AI insights, ask questions (RAG),
#       generate charts, and export a report - all powered by a LangGraph
#       multi-agent workflow using Google Gemini.
# HOW: Streamlit re-runs this whole script top-to-bottom every time the user
#      interacts with a widget. We use st.session_state to "remember" things
#      (like the uploaded DataFrame) between those re-runs, and a sidebar
#      radio button to decide which "page" function to call.

import streamlit as st
import pandas as pd
import plotly.express as px

from tools.dataframe_tools import build_full_profile
from tools.chart_tools import create_chart, SUPPORTED_CHART_TYPES
from agents.insight_agent import run_insight_agent
from agents.rag_agent import run_rag_agent
from agents.chart_agent import run_chart_agent
from graph.workflow import stream_workflow_with_trace
from utils.helpers import StepTimer, format_agent_trace_entry, get_timestamp


# ----------------------------------------------------------------------------
# PAGE CONFIG - must be the first Streamlit command in the script.
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Data Scientist",
    page_icon="📊",
    layout="wide",
)


# ----------------------------------------------------------------------------
# SESSION STATE SETUP
# WHY: Streamlit reruns app.py from top to bottom on every click. Without
#      session_state, we'd lose the uploaded DataFrame the moment the user
#      clicked any other button. Session state is Streamlit's way of
#      remembering data across reruns for the current browser session.
# ----------------------------------------------------------------------------
def initialize_session_state():
    """Sets up every session_state key the app needs, but only if it's missing."""
    defaults = {
        "df": None,
        "dataset_name": None,
        "profiling_result": None,
        "ai_insights": None,
        "chat_history": [],       # bonus: keeps a running log of RAG questions/answers
        "last_chart_figure": None,
        "last_chart_info": None,
        "report_text": None,
        "agent_trace": [],        # bonus: which agent ran, how long it took
    }
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


initialize_session_state()


# ----------------------------------------------------------------------------
# SIDEBAR - navigation + status indicators (bonus feature)
# ----------------------------------------------------------------------------
st.sidebar.title("📊 AI Data Scientist")
st.sidebar.caption("A beginner-friendly Agentic AI demo built with LangGraph + Gemini")

page = st.sidebar.radio(
    "Navigate to:",
    [
        "🏠 Home",
        "📁 Upload Dataset",
        "🔍 Automated Profiling",
        "💡 AI Insights",
        "💬 Ask your Dataset",
        "📈 Visualization",
        "📝 Generate Report",
    ],
)

st.sidebar.divider()
st.sidebar.subheader("Status")
# Status indicators - a simple bonus feature that shows students what state the app is in.
if st.session_state.df is not None:
    st.sidebar.success(f"Dataset loaded: {st.session_state.dataset_name}")
else:
    st.sidebar.warning("No dataset uploaded yet")

if st.session_state.ai_insights is not None:
    st.sidebar.success("AI Insights generated")
else:
    st.sidebar.info("AI Insights not generated yet")


# ----------------------------------------------------------------------------
# PAGE 1: HOME
# ----------------------------------------------------------------------------
def page_home():
    st.title("🤖 AI Data Scientist — Agentic AI Learning Project")
    st.markdown(
        "Welcome! This project is a **teaching tool** that shows how multiple "
        "small AI agents can work together (an *agentic* system) to analyze a "
        "dataset - all orchestrated with **LangGraph** and powered by **Google Gemini**."
    )

    col1, col2 = st.columns(2)

    with col1:
        with st.expander("❓ What is Agentic AI?", expanded=True):
            st.markdown(
                """
                Normally, you send ONE prompt to an LLM and get ONE answer back.

                **Agentic AI** instead breaks a big task into smaller steps, and
                assigns each step to a specialized "agent" (just a function that
                may or may not call an LLM). Agents can:
                - Use **tools** (like pandas or Plotly) to do real work
                - **Hand off** results to the next agent
                - **Make decisions** about what should happen next

                In this app, 5 agents work together: **Profiler → Insight → RAG
                → Chart → Report**.
                """
            )

    with col2:
        with st.expander("🕸️ What is LangGraph?", expanded=True):
            st.markdown(
                """
                **LangGraph** is a Python library for building agents as a
                **graph**: each agent is a *node*, and *edges* define which
                agent runs next.

                Key concepts you'll see in this project's code:
                - **State** — a shared dictionary passed between agents (`graph/state.py`)
                - **Nodes** — each agent is a node (`graph/workflow.py`)
                - **Edges** — fixed connections between nodes
                - **Conditional Edges** — the graph *decides* which node to run
                  next based on the current state (e.g. "did the user ask a question?")
                """
            )

    st.subheader("🏗️ Architecture")
    st.code(
        """
                Upload CSV
                     |
                     v
          +----------------------+
          |   Profiler Agent     |   <- reads the DataFrame with pandas
          |  (Pandas Analysis)   |
          +----------+-----------+
                     |
                     v
          +----------------------+
          |   Insight Agent      |   <- calls Gemini for business insights
          |  (LLM Business Ideas)|
          +----------+-----------+
                     |
          +----------+-----------+   <- conditional edges (a "decision node")
          v                      v
   +--------------+       +--------------+
   |  RAG Agent   |       | Chart Agent  |
   | Answers Qs   |       | Builds charts|
   +------+-------+       +------+-------+
          |                      |
          +----------+-----------+
                     v
          +----------------------+
          |   Report Agent       |   <- collects everything into a report
          +----------------------+
        """,
        language="text",
    )

    with st.expander("🕸️ See the REAL LangGraph structure (auto-generated from our code)"):
        st.caption(
            "This isn't a hand-drawn picture - it's generated directly from the compiled "
            "graph object using `graph.get_graph().draw_mermaid()`. If you change "
            "`graph/workflow.py`, this diagram updates automatically."
        )
        from graph.workflow import build_workflow
        mermaid_source = build_workflow().get_graph().draw_mermaid()
        st.code(mermaid_source, language="text")

    st.info(
        "👉 Start by going to **📁 Upload Dataset** in the sidebar, or try one of "
        "the sample datasets bundled in this project (`data/employees.csv`, "
        "`data/sales.csv`)."
    )


# ----------------------------------------------------------------------------
# PAGE 2: UPLOAD DATASET
# ----------------------------------------------------------------------------
def page_upload():
    st.title("📁 Upload Dataset")
    st.markdown("Upload a CSV file, or quickly load one of the bundled sample datasets.")

    # Quick-load buttons for the sample datasets - handy for a classroom demo.
    sample_col1, sample_col2 = st.columns(2)
    with sample_col1:
        if st.button("⚡ Load sample: employees.csv", use_container_width=True):
            _load_dataframe(pd.read_csv("data/employees.csv"), "employees.csv")
    with sample_col2:
        if st.button("⚡ Load sample: sales.csv", use_container_width=True):
            _load_dataframe(pd.read_csv("data/sales.csv"), "sales.csv")

    st.divider()

    uploaded_file = st.file_uploader("Or upload your own CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        _load_dataframe(df, uploaded_file.name)

    # Show a preview if a dataset is already loaded (from this run or a previous one).
    if st.session_state.df is not None:
        df = st.session_state.df
        st.success(f"Loaded dataset: **{st.session_state.dataset_name}**")

        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        metric_col1.metric("Rows", df.shape[0])
        metric_col2.metric("Columns", df.shape[1])
        metric_col3.metric("Missing Values", int(df.isnull().sum().sum()))
        metric_col4.metric("Duplicate Rows", int(df.duplicated().sum()))

        st.subheader("Preview")
        st.dataframe(df.head(20), use_container_width=True)

        with st.expander("Column Data Types"):
            st.dataframe(df.dtypes.astype(str).rename("Data Type"), use_container_width=True)


def _load_dataframe(df: pd.DataFrame, name: str):
    """
    WHY: Both the file uploader and the "load sample" buttons need to do the
         same thing - save the DataFrame into session_state and reset any
         stale results from a previous dataset.
    WHAT: Stores the new DataFrame and clears out old profiling/insights/etc.
    HOW: Simple session_state assignments.
    """
    st.session_state.df = df
    st.session_state.dataset_name = name
    # Clear old results since they belonged to a different dataset.
    st.session_state.profiling_result = None
    st.session_state.ai_insights = None
    st.session_state.chat_history = []
    st.session_state.last_chart_figure = None
    st.session_state.report_text = None


# ----------------------------------------------------------------------------
# PAGE 3: AUTOMATED PROFILING
# ----------------------------------------------------------------------------
def page_profiling():
    st.title("🔍 Automated Profiling")

    if st.session_state.df is None:
        st.warning("Please upload a dataset first (see '📁 Upload Dataset').")
        return

    df = st.session_state.df

    if st.button("Run Profiler Agent", type="primary"):
        with st.spinner("Profiler Agent is analyzing your dataset..."):
            st.session_state.profiling_result = build_full_profile(df)

    if st.session_state.profiling_result is None:
        st.info("Click 'Run Profiler Agent' to generate the profile below.")
        return

    profile = st.session_state.profiling_result

    st.subheader("Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", profile["shape"]["rows"])
    col2.metric("Columns", profile["shape"]["columns"])
    col3.metric("Duplicate Rows", profile["duplicate_rows"])
    col4.metric("Columns with Missing Values", len(profile["missing_values"]))

    with st.expander("🧩 Missing Values", expanded=True):
        if profile["missing_values"]:
            st.dataframe(pd.Series(profile["missing_values"], name="Missing Count"), use_container_width=True)
        else:
            st.success("No missing values found!")

    with st.expander("🔢 Numeric vs Categorical Columns"):
        st.write("**Numeric columns:**", ", ".join(profile["numeric_columns"]) or "None")
        st.write("**Categorical columns:**", ", ".join(profile["categorical_columns"]) or "None")

    with st.expander("📈 Summary Statistics"):
        if profile["summary_statistics"]:
            st.dataframe(pd.DataFrame(profile["summary_statistics"]), use_container_width=True)
        else:
            st.info("No numeric columns to summarize.")

    with st.expander("🔗 Correlation Matrix"):
        if profile["correlation"]:
            correlation_df = pd.DataFrame(profile["correlation"])
            st.dataframe(correlation_df, use_container_width=True)
            fig = px.imshow(correlation_df, text_auto=True, title="Correlation Heatmap", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Need at least 2 numeric columns to compute correlation.")

    with st.expander("🏷️ Unique Value Counts (Top 5 per column)"):
        if profile["unique_values"]:
            for column, counts in profile["unique_values"].items():
                st.write(f"**{column}**")
                st.dataframe(pd.Series(counts, name="Count"), use_container_width=True)
        else:
            st.info("No categorical columns found.")


# ----------------------------------------------------------------------------
# PAGE 4: AI INSIGHTS
# ----------------------------------------------------------------------------
def page_insights():
    st.title("💡 AI Insights")

    if st.session_state.df is None:
        st.warning("Please upload a dataset first (see '📁 Upload Dataset').")
        return
    if st.session_state.profiling_result is None:
        st.warning("Please run the Profiler Agent first (see '🔍 Automated Profiling').")
        return

    if st.button("✨ Generate Insights", type="primary"):
        timer = StepTimer().start()
        with st.spinner("Insight Agent is thinking (calling Gemini)..."):
            # We build a minimal state dict - the Insight Agent only needs
            # profiling_result to do its job.
            state = {"profiling_result": st.session_state.profiling_result}
            try:
                state = run_insight_agent(state)
                st.session_state.ai_insights = state["ai_insights"]
                duration = timer.stop()
                st.session_state.agent_trace.append(
                    format_agent_trace_entry("Insight Agent", duration)
                )
            except ValueError as error:
                st.error(str(error))

    if st.session_state.ai_insights:
        st.markdown(st.session_state.ai_insights)
        st.download_button(
            "⬇️ Download Insights (Markdown)",
            data=st.session_state.ai_insights,
            file_name="ai_insights.md",
            mime="text/markdown",
        )
    else:
        st.info("Click 'Generate Insights' to have Gemini analyze your dataset.")


# ----------------------------------------------------------------------------
# PAGE 5: ASK YOUR DATASET (RAG)
# ----------------------------------------------------------------------------
def page_ask_dataset():
    st.title("💬 Ask your Dataset")
    st.caption("This page demonstrates RAG: DataFrame -> text -> embeddings -> FAISS -> retrieval -> LLM answer.")

    if st.session_state.df is None:
        st.warning("Please upload a dataset first (see '📁 Upload Dataset').")
        return

    df = st.session_state.df
    numeric_columns = df.select_dtypes(include="number").columns.tolist()

    st.markdown("**Example questions you can try:**")
    example_questions = [
        f"What is the average {numeric_columns[0]}?" if numeric_columns else "What columns does this dataset have?",
        "What trends do you observe in this data?",
        "Summarize this dataset in 2 sentences.",
    ]
    example_cols = st.columns(len(example_questions))
    clicked_example = None
    for col, question in zip(example_cols, example_questions):
        if col.button(question):
            clicked_example = question

    question = st.text_input("Type your question about the dataset:", value=clicked_example or "")

    if st.button("Ask", type="primary") and question:
        timer = StepTimer().start()
        with st.spinner("RAG Agent is retrieving relevant rows and asking Gemini..."):
            state = {"uploaded_dataframe": df, "user_question": question}
            try:
                state = run_rag_agent(state)
                duration = timer.stop()
                st.session_state.chat_history.append({
                    "question": question,
                    "answer": state["rag_answer"],
                    "retrieved": state["retrieved_documents"],
                })
                st.session_state.agent_trace.append(
                    format_agent_trace_entry("RAG Agent", duration)
                )
            except ValueError as error:
                st.error(str(error))

    # Show the chat history, most recent first (bonus feature: chat history).
    for turn in reversed(st.session_state.chat_history):
        st.markdown(f"**🧑 You:** {turn['question']}")
        st.markdown(f"**🤖 AI:** {turn['answer']}")
        with st.expander("🔎 See retrieved context (what the RAG agent found)"):
            for i, chunk in enumerate(turn["retrieved"], start=1):
                st.text(f"Chunk {i}:\n{chunk}")
        st.divider()


# ----------------------------------------------------------------------------
# PAGE 6: VISUALIZATION
# ----------------------------------------------------------------------------
def page_visualization():
    st.title("📈 Visualization")

    if st.session_state.df is None:
        st.warning("Please upload a dataset first (see '📁 Upload Dataset').")
        return

    df = st.session_state.df
    all_columns = df.columns.tolist()

    tab_manual, tab_ai = st.tabs(["🎛️ Build it myself", "🤖 Let the AI recommend a chart"])

    with tab_manual:
        chart_type = st.selectbox("Chart type", SUPPORTED_CHART_TYPES)
        x_axis = st.selectbox("X-axis", all_columns)
        y_axis = None
        if chart_type != "Histogram":
            y_axis = st.selectbox("Y-axis", all_columns, index=min(1, len(all_columns) - 1))

        if st.button("Render Chart", type="primary"):
            try:
                figure = create_chart(df, chart_type, x_axis, y_axis)
                st.session_state.last_chart_figure = figure
                st.session_state.last_chart_info = {
                    "chart_type": chart_type, "x_axis": x_axis, "y_axis": y_axis,
                    "reason": "Selected manually by the user.",
                }
            except Exception as error:
                st.error(f"Could not build this chart: {error}")

    with tab_ai:
        st.markdown("The **Chart Agent** will look at your columns and pick the best chart type + axes.")
        if st.button("🤖 Recommend & Draw Chart", type="primary"):
            timer = StepTimer().start()
            with st.spinner("Chart Agent is deciding the best visualization (calling Gemini)..."):
                state = {"uploaded_dataframe": df, "profiling_result": build_full_profile(df)}
                try:
                    state = run_chart_agent(state)
                    duration = timer.stop()
                    st.session_state.last_chart_figure = state["generated_chart"]
                    st.session_state.last_chart_info = {
                        "chart_type": state["chart_type"], "x_axis": state["chart_x_axis"],
                        "y_axis": state["chart_y_axis"], "reason": state["chart_reason"],
                    }
                    st.session_state.agent_trace.append(
                        format_agent_trace_entry("Chart Agent", duration)
                    )
                except ValueError as error:
                    st.error(str(error))

    if st.session_state.last_chart_figure is not None:
        info = st.session_state.last_chart_info
        st.success(
            f"**Chart:** {info['chart_type']} | **X:** {info['x_axis']} | "
            f"**Y:** {info['y_axis']} | **Why:** {info['reason']}"
        )
        st.plotly_chart(st.session_state.last_chart_figure, use_container_width=True)

        # Bonus: download the chart as a standalone interactive HTML file.
        chart_html = st.session_state.last_chart_figure.to_html()
        st.download_button("⬇️ Download Chart (HTML)", data=chart_html, file_name="chart.html", mime="text/html")


# ----------------------------------------------------------------------------
# PAGE 7: GENERATE REPORT
# ----------------------------------------------------------------------------
def page_report():
    st.title("📝 Generate Report")
    st.caption("This page runs the FULL LangGraph workflow end-to-end, so you can watch every agent run.")

    if st.session_state.df is None:
        st.warning("Please upload a dataset first (see '📁 Upload Dataset').")
        return

    df = st.session_state.df

    with st.expander("⚙️ Optional: include a question and/or chart in the report"):
        include_question = st.text_input("Question to answer in the report (optional)")
        include_chart = st.checkbox("Also generate a chart for the report", value=True)

    if st.button("🚀 Run Full Workflow & Generate Report", type="primary"):
        progress_bar = st.progress(0, text="Starting workflow...")
        status_placeholder = st.empty()
        trace_rows = []

        # Roughly how many nodes we expect to run, just to drive the progress bar.
        total_expected_steps = 5
        step_count = 0

        for node_name, elapsed_seconds, node_state in stream_workflow_with_trace(
            df,
            dataset_name=st.session_state.dataset_name or "dataset.csv",
            user_question=include_question or None,
            chart_requested=include_chart,
        ):
            step_count += 1
            friendly_name = node_name.replace("_", " ").title()
            status_placeholder.info(f"🏃 Currently running: **{friendly_name}**")
            trace_rows.append({"Agent": friendly_name, "Duration (seconds)": elapsed_seconds})
            progress_bar.progress(min(step_count / total_expected_steps, 1.0), text=f"Ran: {friendly_name}")
            final_state = node_state  # keep the latest state - the last one is the complete result

        progress_bar.progress(1.0, text="Workflow complete!")
        status_placeholder.success("✅ All agents finished running.")

        st.session_state.report_text = final_state.get("report")
        st.session_state.agent_trace.extend(trace_rows)

        st.subheader("🕵️ Agent Execution Trace")
        st.dataframe(pd.DataFrame(trace_rows), use_container_width=True)

    if st.session_state.report_text:
        st.subheader("📄 Report Preview")
        st.markdown(st.session_state.report_text)

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "⬇️ Download Report (Markdown)",
                data=st.session_state.report_text,
                file_name=f"report_{get_timestamp().replace(':', '-').replace(' ', '_')}.md",
                mime="text/markdown",
            )
        with col2:
            pdf_bytes = _markdown_report_to_pdf_bytes(st.session_state.report_text)
            st.download_button(
                "⬇️ Download Report (PDF)",
                data=pdf_bytes,
                file_name=f"report_{get_timestamp().replace(':', '-').replace(' ', '_')}.pdf",
                mime="application/pdf",
            )


def _sanitize_text_for_pdf(text: str) -> str:
    """
    WHY: fpdf2's built-in "Helvetica" font only supports the Latin-1 character
         set. Gemini's generated text often contains Unicode punctuation
         (smart quotes, en/em dashes, bullet points, checkmarks) that Latin-1
         cannot represent - and fpdf2 crashes instead of skipping them.
    WHAT: Swaps common "smart" Unicode characters for plain ASCII equivalents,
          then replaces anything else outside Latin-1 with "?" as a safety net.
    HOW: A small lookup table for the common cases, then `.encode("latin-1",
         "replace")` catches everything we didn't think of.
    """
    unicode_to_ascii = {
        "‘": "'", "’": "'",      # smart single quotes
        "“": '"', "”": '"',      # smart double quotes
        "–": "-", "—": "-",      # en dash, em dash
        "•": "-",                     # bullet point
        "…": "...",                   # ellipsis
        "✅": "[x]", "✔": "[x]", "☑": "[x]",  # checkmarks
    }
    for unicode_char, ascii_char in unicode_to_ascii.items():
        text = text.replace(unicode_char, ascii_char)

    # Anything still outside Latin-1 (e.g. emoji) becomes "?" instead of
    # crashing the PDF renderer.
    return text.encode("latin-1", "replace").decode("latin-1")


def _markdown_report_to_pdf_bytes(markdown_text: str) -> bytes:
    """
    WHY: Some students/teachers prefer a PDF over a markdown file.
    WHAT: Converts the plain report text into a simple PDF using fpdf2.
    HOW: fpdf2 can only write plain text easily, so we strip markdown symbols
         and print each line - simple and good enough for a teaching project.
    """
    from fpdf import FPDF
    from fpdf.enums import XPos, YPos

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)

    for line in markdown_text.split("\n"):
        clean_line = line.replace("#", "").replace("**", "").replace("*", "-")
        clean_line = _sanitize_text_for_pdf(clean_line)
        if clean_line.strip():
            # WHY: fpdf2 leaves the cursor at the RIGHT edge of the text by
            # default after multi_cell(). Without resetting it back to the
            # left margin here, the NEXT line would have almost no width
            # left to render into and fpdf2 would crash.
            pdf.multi_cell(0, 8, clean_line, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        else:
            # Blank line - just move down instead of rendering an empty cell.
            pdf.ln(4)

    return bytes(pdf.output())


# ----------------------------------------------------------------------------
# ROUTER - calls the right page function based on the sidebar selection.
# ----------------------------------------------------------------------------
page_router = {
    "🏠 Home": page_home,
    "📁 Upload Dataset": page_upload,
    "🔍 Automated Profiling": page_profiling,
    "💡 AI Insights": page_insights,
    "💬 Ask your Dataset": page_ask_dataset,
    "📈 Visualization": page_visualization,
    "📝 Generate Report": page_report,
}

page_router[page]()

# Bonus: show the full agent execution trace (across all pages) at the bottom of the sidebar.
if st.session_state.agent_trace:
    with st.sidebar.expander("🕵️ Agent Execution Trace"):
        st.dataframe(pd.DataFrame(st.session_state.agent_trace), use_container_width=True, hide_index=True)
