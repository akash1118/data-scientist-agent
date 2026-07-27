# llmops/token_manager.py
# --------------------------
# WHY: LLM providers charge (and rate-limit) based on TOKENS, not characters
#      or words. A "token" is roughly 3/4 of an English word. Understanding
#      and tracking token usage is the first step toward controlling cost
#      and staying under rate limits.
# WHAT: A function to pull token counts out of an LLM response, and a small
#       tracker class that keeps a running log + totals across many calls.
# HOW: Modern LangChain chat models (Gemini, Groq, etc.) attach a
#      `usage_metadata` dictionary to every AIMessage response automatically -
#      we just read it. No manual counting needed.

from datetime import datetime


def extract_token_usage(llm_response) -> dict:
    """
    WHY: Every agent that calls an LLM (Insight, RAG, Chart) should record
         how many tokens that call used, and this is the ONE place that
         knows how to read that off a LangChain response object.
    WHAT: Returns {"input_tokens", "output_tokens", "total_tokens", "model_name"}.
    HOW: Reads `response.usage_metadata` (token counts) and
         `response.response_metadata["model_name"]` (which model actually
         answered - useful since a Gemini call might have silently failed
         over to Groq). Falls back to zeros/"unknown" if a provider doesn't
         report this, so the app never crashes because of missing metadata.
    """
    usage = getattr(llm_response, "usage_metadata", None) or {}
    metadata = getattr(llm_response, "response_metadata", None) or {}

    return {
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
        "model_name": metadata.get("model_name", "unknown"),
    }


class TokenUsageTracker:
    """
    WHY: A single LLM call's token count isn't very interesting on its own -
         what matters for cost/rate-limit awareness is the RUNNING TOTAL
         across an entire session (or app lifetime).
    WHAT: Keeps a list of every recorded usage event, plus running totals.
    HOW: Call .record(agent_name, usage_dict) after every LLM call - usage_dict
         is whatever extract_token_usage() returned (it already includes
         model_name). Read .history for the raw log, or .totals() for the summary.
    """

    def __init__(self):
        self.history = []

    def record(self, agent_name: str, usage: dict) -> dict:
        """WHAT: Appends one usage event to the history and returns it."""
        entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "agent": agent_name,
            "model": usage.get("model_name", "unknown"),
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }
        self.history.append(entry)
        return entry

    def totals(self) -> dict:
        """WHAT: Returns the running totals across every recorded call so far."""
        return {
            "total_calls": len(self.history),
            "input_tokens": sum(e["input_tokens"] for e in self.history),
            "output_tokens": sum(e["output_tokens"] for e in self.history),
            "total_tokens": sum(e["total_tokens"] for e in self.history),
        }
