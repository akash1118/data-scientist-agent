# llmops/__init__.py
# --------------------
# WHY: This package is a SEPARATE teaching module that demonstrates common
#      "LLMOps" / production AI-engineering concepts on top of the main
#      LangGraph app, without touching the core agents/graph/tools code.
# WHAT: One small, beginner-friendly file per concept:
#         - model_serving.py  -> wraps our agents as reusable "service" functions
#         - api_server.py     -> a minimal FastAPI app exposing those functions
#         - token_manager.py  -> counts input/output tokens per LLM call
#         - caching.py        -> caches identical LLM calls so repeats are instant
#         - monitoring.py     -> logs every agent run (latency, success/failure)
#         - evaluation.py     -> scores how good an AI answer is
#         - guardrails.py     -> checks user input / LLM output for safety issues
#         - cost_optimizer.py -> estimates $ cost per call and gives savings tips
# HOW: See app.py's "🏭 LLMOps & Production" page for a live, interactive demo
#      of every concept in this package.
