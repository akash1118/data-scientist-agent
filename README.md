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
| 🏭 LLMOps & Production | Hands-on demos of Model Serving, an API, Token Management, Caching, Monitoring, Evaluation, Guardrails, and Cost Optimization |

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
├── app.py                  # The ENTIRE Streamlit UI (all 8 pages, sidebar nav)
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
├── llmops/                  # Beginner-friendly production/LLMOps concepts (see below)
│   ├── model_serving.py     # Wraps agents as reusable "service" functions
│   ├── api_server.py        # Minimal FastAPI app exposing those functions over HTTP
│   ├── token_manager.py     # Reads token usage off every LLM response
│   ├── caching.py           # Turns on LangChain's SQLite-backed LLM cache
│   ├── monitoring.py        # Logs every agent run (latency, success/failure)
│   ├── evaluation.py        # Groundedness check + LLM-as-judge scoring
│   ├── guardrails.py        # Input/output validation checks
│   └── cost_optimizer.py    # Token -> estimated $ cost, plus savings tips
│
├── utils/
│   ├── llm.py                # Builds the chat model (Gemini + Groq fallback) + local embeddings
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

### 5. (Optional) Add Groq as an automatic fallback

Gemini is the **primary** chat provider, but you can wire in **Groq** (fast,
cheap, open-source models) as an automatic **fallback**: if a call to Gemini
fails for any reason (rate limit, quota exceeded, network error, bad key...),
LangChain automatically retries the SAME request against Groq instead of
crashing the app. Get a free key at **https://console.groq.com/keys**, then
in your `.env`:

```
GROQ_API_KEY=your_real_groq_key_here
GROQ_MODEL=llama-3.1-8b-instant
```

Leave `GROQ_API_KEY` blank (or as the placeholder) to disable this - the app
will just use Gemini alone. See `utils/llm.py -> get_llm()`, which uses
LangChain's `.with_fallbacks([...])` to wire this up in a few lines.

Embeddings (used by the RAG page) don't need either key - see the next
section.

### 6. (Optional) Regenerate the sample datasets

Two datasets are already included in `data/`, but you can regenerate them
any time:

```bash
python scripts/generate_sample_data.py
```

### 7. Run the app

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
   chunk is converted into a number vector (embedding) using a small,
   **local, open-source** model (`sentence-transformers/all-MiniLM-L6-v2`,
   see `utils/llm.py -> get_embedding_model()`), and stored in a FAISS index
   for fast similarity search. This step needs **no API key at all** - it
   runs entirely on your own machine.
4. **Retrieval** (`rag/vector_store.retrieve_relevant_chunks`) — the user's
   question is embedded the same way, and FAISS returns the top-k most
   similar chunks.
5. **Answering** — those chunks are stuffed into a prompt, and the chat LLM
   (Gemini, or Groq if Gemini failed - see below) answers using _only_ that
   context.

Try asking:

- "What is the average salary?"
- "Which department has the highest revenue?" _(use sales.csv)_
- "Show employees above age 30"
- "What trends do you observe?"

---

## 🏭 LLMOps & Production Concepts

Building a working demo is step one - running it for real users takes more
engineering. The **🏭 LLMOps & Production** page is a hands-on tour of eight
concepts that turn a demo into something production-worthy, each in its own
small file under `llmops/` and each with a live, interactive demo in the UI.

| Concept                | File                        | What the demo does                                                                          |
| ----------------------- | --------------------------- | --------------------------------------------------------------------------------------------- |
| **Model Serving**       | `llmops/model_serving.py`   | Wraps the Profiler/Insight/RAG agents as plain, reusable functions - no Streamlit, no LangGraph state, just data in → data out. |
| **API Development**     | `llmops/api_server.py`      | A minimal FastAPI app (`/health`, `/datasets`, `/profile`, `/insights`, `/ask`) built on the same serving functions. The demo tab launches it in a **background thread inside the Streamlit process itself**, then calls it live over real HTTP. |
| **Token Management**    | `llmops/token_manager.py`   | Every agent call automatically records `input_tokens` / `output_tokens` / `total_tokens`, read straight off LangChain's `response.usage_metadata`. |
| **Caching**             | `llmops/caching.py`         | Turns on LangChain's global LLM cache (`set_llm_cache`), backed by a local SQLite file. The demo runs the same prompt twice so you can see the ~1000x+ speedup on the second (cached) call. |
| **Monitoring**          | `llmops/monitoring.py`      | Logs every agent run's latency and success/failure, both in-memory (for a live dashboard) and to `reports/monitoring_log.csv` (so it survives restarts). |
| **Evaluation**          | `llmops/evaluation.py`      | Two techniques: a free/instant **groundedness** check (does the answer's wording overlap with the retrieved context?) and an **LLM-as-judge** (a second LLM call grades the first one's answer 1-5). |
| **Guardrails**          | `llmops/guardrails.py`      | Input checks (empty, too long, looks like prompt injection) run **before** the LLM call; output checks (empty, looks like it contains sensitive data) run **after**. These same checks are wired into the real "Ask your Dataset" page, not just the demo tab. |
| **Cost Optimization**   | `llmops/cost_optimizer.py`  | Converts each call's token counts into an estimated USD cost using an illustrative per-model pricing table, plus a static list of concrete cost-saving tips. |

Everything on this page is **beginner-friendly by design**: no external
monitoring/observability service, no real payment processor, no production
API gateway - just the core idea behind each concept, implemented from
scratch in well under 100 lines per file.

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
✔ LLM Integration (Google Gemini + Groq fallback) &nbsp; ✔ Docker
✔ Model Serving &nbsp; ✔ API Development (FastAPI) &nbsp; ✔ Token Management
✔ LLM Response Caching &nbsp; ✔ Monitoring &nbsp; ✔ Evaluation (incl. LLM-as-judge)
✔ Guardrails &nbsp; ✔ Cost Optimization

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
(`gemini-2.5-flash` by default, configurable) · Groq (optional fallback,
`llama-3.1-8b-instant`) · Sentence-Transformers (local embeddings,
`all-MiniLM-L6-v2`) · Pandas · Matplotlib · Plotly · FAISS · FastAPI + uvicorn
(LLMOps demo API only) · Docker · python-dotenv

## ⚠️ Notes on Cost & API Limits

Google AI Studio offers a free tier for Gemini. Generating insights, chart
recommendations, and RAG answers each make one LLM call; running the full
workflow on the **Generate Report** page makes 2–3 calls total. Keep an eye
on your usage if you're demoing this to a large class simultaneously.
