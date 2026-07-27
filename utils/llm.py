# utils/llm.py
# -------------
# WHY: Every agent in this project needs to talk to an LLM. Instead of
#      repeating the same setup code in five different agent files, we
#      centralize it here in ONE place. If we ever want to swap models -
#      or add resilience against a provider going down - we only change
#      this file.
# WHAT: Creates the chat model our agents use (Google Gemini as the primary
#       provider, with Groq wired in as an automatic FALLBACK if Gemini
#       fails), plus a small open-source embedding model that runs locally
#       for the RAG agent (no API key needed at all for embeddings).
# HOW:  We read API keys/model names from environment variables (loaded from
#       your .env file by python-dotenv), then build and return
#       LangChain-compatible objects the rest of the app uses. Every agent
#       calls get_llm() and never imports a specific provider directly.

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load variables from the .env file into the environment (os.environ).
# This must run before we try to read any keys below.
load_dotenv()


def _is_real_key(value: str) -> bool:
    """WHY: .env.example ships with placeholder text like 'your_..._here' -
    this treats that placeholder the same as "not set" so we don't try to
    call an API with an obviously fake key."""
    return bool(value) and not value.startswith("your_")


def _groq_is_configured() -> bool:
    """WHAT: True only if the user has actually filled in a Groq API key."""
    return _is_real_key(os.getenv("GROQ_API_KEY", ""))


def _get_gemini_llm(temperature: float) -> ChatGoogleGenerativeAI:
    """
    WHY: This is the PRIMARY provider for the whole project.
    WHAT: Builds a Google Gemini chat model.
    HOW: Reads GOOGLE_API_KEY and GEMINI_MODEL from the environment.
    """
    api_key = os.getenv("GOOGLE_API_KEY", "")
    if not _is_real_key(api_key):
        raise ValueError(
            "GOOGLE_API_KEY is missing. Please copy .env.example to .env "
            "and paste your Google Gemini API key from https://aistudio.google.com/app/apikey"
        )

    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=temperature,
    )


def _get_groq_llm(temperature: float):
    """
    WHY: Groq runs open-source models (like Llama) on very fast, very cheap
         hardware. We use it as a FALLBACK: if Gemini is down, rate-limited,
         or misconfigured, requests automatically retry against Groq instead
         of crashing the whole app.
    WHAT: Builds a Groq chat model using LangChain's "langchain-groq" package.
    HOW: Reads GROQ_API_KEY and GROQ_MODEL from the environment. We import
         langchain_groq HERE (not at the top of the file) so that students
         who never configure Groq don't need the package installed at all.
    """
    from langchain_groq import ChatGroq

    api_key = os.getenv("GROQ_API_KEY", "")

    # "llama-3.1-8b-instant" is one of Groq's cheapest and fastest models -
    # a great match for a classroom demo where cost/speed matter more than
    # top-tier reasoning quality.
    model_name = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

    return ChatGroq(
        model=model_name,
        api_key=api_key,
        temperature=temperature,
    )


def get_llm(temperature: float = None):
    """
    WHY: All 4 of our agents that "think" (Insight, RAG, Chart, Report) need
         a chat model. This function is the single place that builds it, and
         wires up automatic failover so a Gemini outage doesn't break the demo.
    WHAT: Returns a LangChain chat model. If a Groq API key is configured,
          the returned object is Gemini WRAPPED with Groq as a fallback -
          LangChain will silently retry with Groq if a call to Gemini raises
          an error (rate limit, quota exceeded, network issue, bad key, etc).
          If Gemini itself can't even be built (e.g. GOOGLE_API_KEY missing)
          and Groq IS configured, we skip straight to Groq as the primary.
    HOW: Uses LangChain's built-in `.with_fallbacks([...])` - a Runnable
         method that wraps a chain/model with one or more backup models to
         try, in order, if the first one raises an exception.
    """
    if temperature is None:
        temperature = float(os.getenv("LLM_TEMPERATURE", "0.3"))

    groq_configured = _groq_is_configured()

    try:
        primary_llm = _get_gemini_llm(temperature)
    except ValueError:
        # Gemini isn't configured at all. If Groq is available, use it
        # directly instead of crashing the app.
        if groq_configured:
            return _get_groq_llm(temperature)
        # Neither provider is configured - re-raise the original, helpful error.
        raise

    if groq_configured:
        # Gemini is our primary model, but LangChain will automatically retry
        # with Groq if any call to Gemini raises an exception.
        fallback_llm = _get_groq_llm(temperature)
        return primary_llm.with_fallbacks([fallback_llm])

    # No Groq key configured - just use Gemini, no fallback available.
    return primary_llm


def get_embedding_model():
    """
    WHY: The RAG Agent (Page 5, "Ask your Dataset") needs to convert text
         into number vectors (embeddings) so it can search for similar text.
         We use a SMALL OPEN-SOURCE embedding model that runs locally on your
         own machine, instead of calling a paid API. This means: no API key
         needed for embeddings, no per-call cost, and it works even if both
         Gemini and Groq are unavailable.
    WHAT: Returns a HuggingFace "sentence-transformers" embedding model.
          The default, "all-MiniLM-L6-v2", is a well-known, compact model
          (~90MB, 384-dimensional vectors) that's a common teaching example
          for lightweight local embeddings.
    HOW: LangChain's HuggingFaceEmbeddings wraps the `sentence-transformers`
         library. The first time this runs, it downloads the model from the
         HuggingFace Hub and caches it locally (~/.cache) - after that, it
         works fully offline.
    """
    from langchain_huggingface import HuggingFaceEmbeddings

    model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    # model_kwargs={"device": "cpu"} keeps this simple and portable - it runs
    # on any machine without needing a GPU or CUDA setup.
    return HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"device": "cpu"})
