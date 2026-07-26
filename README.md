# 🤖 AI Data Scientist — A Beginner-Friendly Agentic AI Project

A teaching-focused Streamlit app that shows how **multiple small AI agents**
can work together (an _agentic system_) to analyze a CSV dataset — profiling
it, generating insights, answering questions (RAG), building charts, and
producing a report — all orchestrated with **LangGraph** and powered by
**Google Gemini**.

Built for students who know basic Python and basic Generative AI concepts,
and want to understand Agentic AI, LangGraph, RAG, and multi-agent systems
by reading real, working, heavily-commented code.

---

## ✨ What You Can Do

| Page                   | What it demonstrates                                                                    |
| ---------------------- | --------------------------------------------------------------------------------------- |
| 🏠 Home                | What Agentic AI and LangGraph are, plus the real graph diagram                          |
| 📁 Upload Dataset      | CSV upload + instant preview                                                            |
| 🔍 Automated Profiling | Pandas-based data profiling (shape, missing values, stats, correlation)                 |
| 💡 AI Insights         | The **Insight Agent** calls Gemini to write a business analysis                         |
| 💬 Ask your Dataset    | A full **RAG** pipeline: DataFrame → text → embeddings → FAISS → retrieval → LLM answer |
| 📈 Visualization       | The **Chart Agent** picks the best chart type + axes and explains why                   |
| 📝 Generate Report     | Runs the **entire LangGraph workflow** end-to-end and exports Markdown/PDF              |

---

## 🏗️ Architecture

```
                Upload CSV
                     │
                     ▼
          ┌─────────────────────┐
          │  Profiler Agent      │   <- pandas only, no LLM
          │ (Pandas Analysis)    │
          └──────────┬───────────┘
                     │
                     ▼
          ┌─────────────────────┐
          │ Insight Agent        │   <- calls Gemini
          │ (LLM Business Ideas) │
          └──────────┬───────────┘
                     │
         ┌───────────┴────────────┐   <- conditional edge (a "decision node")
         ▼                        ▼
 ┌─────────────────┐     ┌─────────────────┐
 │ RAG Agent       │     │ Chart Agent     │
 │ Answer Questions│     │ Create Charts   │
 └────────┬────────┘     └────────┬────────┘
          └────────────┬──────────┘
                       ▼
             ┌──────────────────┐
             │ Report Agent     │
             └──────────────────┘
                       │
                       ▼
                      END
```

This is a **real** LangGraph `StateGraph` — the exact diagram is also
auto-generated from the compiled graph object and shown on the Home page
(`graph.get_graph().draw_mermaid()`), so it can never drift out of sync with
the code.

### Why this shape?

Instead of many hyper-specialized agents, the workflow is intentionally kept
to **5 agents with 2 decision points**, which is enough to demonstrate every
core LangGraph concept (state, nodes, edges, conditional edges, tool
calling) without overwhelming a beginner.

---

## 📂 Project Structure

```
AI_Data_Analyst/
├── app.py                  # The ENTIRE Streamlit UI (all 7 pages, sidebar nav)
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
│
├── agents/                 # One file per agent - the "thinking" units
│   ├── profiler_agent.py   # Pandas profiling (no LLM)
│   ├── insight_agent.py    # Gemini-generated business insights
│   ├── rag_agent.py        # Retrieval-Augmented Generation Q&A
│   ├── chart_agent.py      # Gemini recommends chart type + axes
│   └── report_agent.py     # Assembles the final markdown report
│
├── graph/                  # The LangGraph wiring
│   ├── state.py             # AgentState - the shared dictionary passed between agents
│   └── workflow.py          # StateGraph definition: nodes, edges, conditional edges
│
├── tools/                  # Plain Python functions the agents call (no LLM)
│   ├── dataframe_tools.py   # pandas profiling helpers
│   ├── chart_tools.py       # Plotly chart builder
│   └── rag_tools.py         # DataFrame -> text documents -> chunks
│
├── rag/
│   └── vector_store.py      # FAISS build + retrieve functions
│
├── utils/
│   ├── llm.py                # Builds the Gemini chat + embedding models
│   └── helpers.py            # Timestamps, timers, folder helpers
│
├── scripts/
│   └── generate_sample_data.py  # Regenerates data/employees.csv and data/sales.csv
│
├── data/                   # Bundled sample datasets
│   ├── employees.csv        # ~200 rows: name, age, department, salary, ...
│   └── sales.csv             # ~200 rows: product, revenue, profit, region, ...
│
├── reports/                # Generated reports are saved here
└── uploads/                 # (reserved for uploaded files, if you extend the app)
```

---

## 🚀 Getting Started (Local)

### 1. Get a free Google Gemini API key

Go to **https://aistudio.google.com/app/apikey** and create a free API key.

### 2. Create a virtual environment

```bash
python3 -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure your API key

```bash
cp .env.example .env
```

Then open `.env` and paste your key:

```
GOOGLE_API_KEY=your_real_key_here
```

### 5. (Optional) Regenerate the sample datasets

Two datasets are already included in `data/`, but you can regenerate them
any time:

```bash
python scripts/generate_sample_data.py
```

### 6. Run the app

```bash
streamlit run app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`) and start
with **📁 Upload Dataset** — you can click "Load sample: employees.csv" to
try the app instantly without your own file.

---

## 🐳 Running with Docker

### Option A: `docker build` + `docker run`

```bash
docker build -t ai-data-analyst .
docker run -p 8501:8501 --env-file .env ai-data-analyst
```

### Option B: `docker compose up`

```bash
docker compose up --build
```

Either way, open **http://localhost:8501**. Make sure you've created a
`.env` file first (see step 4 above) — Docker reads your Gemini API key from it.

---

## 🕸️ How LangGraph Works Here

LangGraph models an agent workflow as a **graph**:

- **State** (`graph/state.py`) — a shared `TypedDict` (`AgentState`) passed
  from agent to agent. Each agent reads a few fields and writes a few new ones.
- **Nodes** (`graph/workflow.py`) — each agent function (`run_profiler_agent`,
  `run_insight_agent`, ...) is registered as a node with `workflow.add_node(...)`.
- **Edges** — fixed connections, e.g. `workflow.add_edge("profiler_agent", "insight_agent")`
  always runs Insight right after Profiler.
- **Conditional Edges** — the graph _decides_ what runs next based on the
  current state. After Insight, `route_after_insight()` checks: did the user
  ask a question? Request a chart? Neither? — and routes to RAG, Chart, or
  straight to the Report.
- **Tool Calling** — the Chart Agent asks Gemini to _decide_ the best chart
  type and axes (returned as JSON), then a plain Python function
  (`tools/chart_tools.create_chart`) _executes_ that decision by drawing the
  chart. This separation of "decide" (LLM) vs "do" (tool) is the essence of
  tool calling.
- **Messages** — every agent appends a short log entry to `state["messages"]`,
  giving you a running trace of what happened.
- **Streaming** — `graph/workflow.py` also has `stream_workflow_with_trace()`,
  which uses `graph.stream(..., stream_mode="updates")` so the Streamlit UI
  can show "which agent is currently running" live, on the **Generate
  Report** page.

Checkpointing (saving/resuming graph runs) is a real LangGraph feature but is
intentionally left out here to keep the project focused — see LangGraph's
docs on `MemorySaver` if you want to add it as an exercise.

---

## 🔍 How the RAG Pipeline Works (Page: "Ask your Dataset")

RAG stands for **Retrieval-Augmented Generation** — instead of asking the LLM
to "just know" the answer, we first _retrieve_ the most relevant facts, and
then ask the LLM to answer _using only that retrieved context_. This reduces
hallucination and lets the LLM answer questions about data it was never
trained on.

Step by step (see `agents/rag_agent.py`):

1. **DataFrame → text** (`tools/rag_tools.dataframe_to_documents`) — every
   row becomes a sentence, e.g.
   `"Employee Name is John Smith. Department is Sales. Salary is 45000."`
2. **Chunking** (`tools/rag_tools.chunk_documents`) — rows are grouped into
   small chunks so the LLM gets a bit more context per lookup.
3. **Embeddings + FAISS** (`rag/vector_store.build_vector_store`) — every
   chunk is converted into a number vector (embedding) using Gemini's
   embedding model, and stored in a FAISS index for fast similarity search.
4. **Retrieval** (`rag/vector_store.retrieve_relevant_chunks`) — the user's
   question is embedded the same way, and FAISS returns the top-k most
   similar chunks.
5. **Answering** — those chunks are stuffed into a prompt, and Gemini answers
   using _only_ that context.

Try asking:

- "What is the average salary?"
- "Which department has the highest revenue?" _(use sales.csv)_
- "Show employees above age 30"
- "What trends do you observe?"

---

## 🎓 Teaching Notes

This project intentionally favors **readability over performance**:

- Every file starts with a comment block explaining its purpose.
- Every function has comments explaining **why**, **what**, and **how**.
- No advanced Python tricks (no metaclasses, no complex decorators, minimal
  type-hint gymnastics).
- Functions are kept short and single-purpose.
- The "decide vs do" split (LLM decides, plain functions execute) is used
  consistently in the Chart Agent — a core agentic-AI pattern worth
  highlighting in class.

### Suggested classroom flow (2–3 hours)

1. Run the app and click through all 7 pages (15 min).
2. Read `graph/state.py` then `graph/workflow.py` together — this is the
   "spine" of the whole project (30 min).
3. Read one simple agent (`agents/profiler_agent.py`) and one LLM-powered
   agent (`agents/insight_agent.py`) (30 min).
4. Walk through the RAG pipeline in `agents/rag_agent.py` +
   `rag/vector_store.py` (30 min).
5. Have students modify something small: add a new example question, add a
   new chart type to `tools/chart_tools.py`, or add a new profiling metric to
   `tools/dataframe_tools.py` (30–45 min).

### Concepts covered

✔ Streamlit &nbsp; ✔ LangGraph &nbsp; ✔ StateGraph &nbsp; ✔ Multi-Agent Systems
✔ Tool Calling &nbsp; ✔ RAG &nbsp; ✔ FAISS &nbsp; ✔ Pandas &nbsp; ✔ Plotly
✔ LLM Integration (Google Gemini) &nbsp; ✔ Docker

---

## 🖼️ Screenshots

_(Add your own screenshots here after running the app — placeholders below.)_

- `docs/screenshot-home.png` — Home page with architecture diagram
- `docs/screenshot-profiling.png` — Automated Profiling metrics
- `docs/screenshot-insights.png` — AI-generated insights
- `docs/screenshot-rag.png` — Ask your Dataset, with retrieved context shown
- `docs/screenshot-chart.png` — AI-recommended chart
- `docs/screenshot-report.png` — Live agent execution trace + final report

---

## 🧰 Tech Stack

Python 3.12 · Streamlit · LangGraph · LangChain · Google Gemini
(`gemini-2.5-flash` by default, configurable) · Pandas · Matplotlib · Plotly ·
FAISS · Docker · python-dotenv

## ⚠️ Notes on Cost & API Limits

Google AI Studio offers a free tier for Gemini. Generating insights, chart
recommendations, and RAG answers each make one LLM call; running the full
workflow on the **Generate Report** page makes 2–3 calls total. Keep an eye
on your usage if you're demoing this to a large class simultaneously.
