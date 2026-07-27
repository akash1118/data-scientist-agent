# llmops/cost_optimizer.py
# ---------------------------
# WHY: Tokens (see token_manager.py) are the unit LLM providers bill you in.
#      Turning "tokens used" into "dollars spent" makes cost REAL and
#      comparable, and highlights which agent/provider is most expensive.
# WHAT: A small price-per-model lookup table plus a function that turns a
#       token_usage dict into an estimated cost in USD, and a static list of
#       common cost-saving techniques for students to try.
# HOW: Plain arithmetic - (tokens / 1,000,000) * price_per_million.

# Illustrative, approximate prices in USD per 1 MILLION tokens.
# WHY these numbers: they're for TEACHING relative cost differences (e.g.
# "see how much cheaper the small model is"), not for real billing decisions.
# Always check the provider's official pricing page for current rates.
PRICING_PER_MILLION_TOKENS = {
    "gemini-2.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-2.5-pro": {"input": 1.25, "output": 5.00},
    "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
    "sentence-transformers/all-MiniLM-L6-v2": {"input": 0.0, "output": 0.0},  # runs locally - free
}

# A safe fallback for any model name we don't have exact pricing for, so the
# demo never crashes on an unrecognized model - it just estimates using a
# rough "mid-range small model" price instead.
DEFAULT_PRICING = {"input": 0.10, "output": 0.30}


def estimate_cost_usd(model_name: str, input_tokens: int, output_tokens: int) -> float:
    """
    WHY: This is the function every other cost display in the app calls.
    WHAT: Returns an estimated cost in US dollars for one LLM call.
    HOW: Looks up the model's price-per-million-tokens, then scales it by
         how many tokens were actually used.
    """
    pricing = PRICING_PER_MILLION_TOKENS.get(model_name, DEFAULT_PRICING)

    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]

    return round(input_cost + output_cost, 6)


def get_optimization_tips() -> list:
    """
    WHY: Cost awareness is only useful if paired with concrete ways to
         reduce it - this is the "so what do I do about it?" payoff.
    WHAT: Returns a static list of common, practical LLM cost-saving tips.
    """
    return [
        "Enable caching (see the Caching tab) - identical prompts cost $0 after the first call.",
        "Use a smaller/cheaper model for simple tasks (e.g. llama-3.1-8b-instant instead of a large model).",
        "Keep prompts short - trim unused context, summarize instead of pasting raw data when possible.",
        "Set a max output length where your provider supports it, so the model can't ramble expensively.",
        "Batch related questions into one call instead of many small separate calls, when it makes sense.",
        "Run embeddings locally (like this project's HuggingFace embeddings) instead of paying per-embedding-call.",
    ]
