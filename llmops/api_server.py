# llmops/api_server.py
# -----------------------
# WHY: "API Development" means exposing your app's logic over HTTP so OTHER
#      programs (not just our Streamlit UI) can use it - a mobile app, a
#      teammate's script, another website. FastAPI is the standard,
#      beginner-friendly way to do this in Python.
# WHAT: A small REST API with 4 endpoints, all backed by the SAME functions
#       from llmops/model_serving.py that the Streamlit demo page also calls -
#       proof that "serving" logic should live in one reusable place.
# HOW: Each endpoint validates its input using a Pydantic model (FastAPI does
#      this automatically), calls a model_serving function, and returns JSON.
#
# Run it standalone with:
#     uvicorn llmops.api_server:app --reload --port 8000
# ...or launch it from inside the Streamlit app - see the "API Development"
# tab on the "LLMOps & Production" page, which starts this same app in a
# background thread so you can call it live without leaving the browser.

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from llmops.model_serving import serve_profile, serve_insights, serve_ask, AVAILABLE_DATASETS

app = FastAPI(
    title="AI Data Analyst API",
    description="A minimal teaching API exposing the same agents used by the Streamlit app.",
    version="1.0.0",
)


# --- Request body schemas ---------------------------------------------------
# WHY: FastAPI uses these Pydantic models to automatically validate incoming
#      JSON (e.g. reject a request that's missing "dataset") before our code
#      ever runs - this is "API Development" 101: never trust raw input.
class DatasetRequest(BaseModel):
    dataset: str  # must be one of AVAILABLE_DATASETS - checked inside the handler


class AskRequest(BaseModel):
    dataset: str
    question: str


# --- Endpoints ---------------------------------------------------------------
@app.get("/health")
def health_check():
    """WHY: The very first endpoint any real API needs - "is this thing alive?"."""
    return {"status": "ok", "message": "AI Data Analyst API is running."}


@app.get("/datasets")
def list_datasets():
    """WHAT: Lists which bundled sample datasets the API can analyze."""
    return {"datasets": AVAILABLE_DATASETS}


@app.post("/profile")
def profile_endpoint(request: DatasetRequest):
    """WHAT: Runs the Profiler Agent (pandas only, no LLM - fast) over a bundled dataset."""
    try:
        return serve_profile(request.dataset)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))


@app.post("/insights")
def insights_endpoint(request: DatasetRequest):
    """WHAT: Runs the Profiler + Insight Agents (calls the LLM) over a bundled dataset."""
    try:
        return serve_insights(request.dataset)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))


@app.post("/ask")
def ask_endpoint(request: AskRequest):
    """WHAT: Runs the RAG Agent - answers a natural-language question about a bundled dataset."""
    try:
        return serve_ask(request.dataset, request.question)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
