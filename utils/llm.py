# utils/llm.py
# -------------
# WHY: Every agent in this project needs to talk to an LLM. Instead of
#      repeating the same setup code in five different agent files, we
#      centralize it here in ONE place. If we ever want to swap models,
#      we only change this file.
# WHAT: Creates a Google Gemini chat model and a Gemini embedding model,
#       using the LangChain "langchain-google-genai" package.
# HOW:  We read the API key and model name from environment variables
#       (loaded from your .env file by python-dotenv), then build and
#       return LangChain-compatible objects that the rest of the app uses.

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Load variables from the .env file into the environment (os.environ).
# This must run before we try to read GOOGLE_API_KEY below.
load_dotenv()


def get_llm(temperature: float = None) -> ChatGoogleGenerativeAI:
    """
    WHY: All 4 of our agents that "think" (Insight, RAG, Chart, Report) need
         a chat model. This function is the single place that builds it.
    WHAT: Returns a ready-to-use Google Gemini chat model object.
    HOW: Reads GOOGLE_API_KEY and GEMINI_MODEL from the environment, and
         falls back to sensible defaults if they are missing.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key == "your_google_gemini_api_key_here":
        raise ValueError(
            "GOOGLE_API_KEY is missing. Please copy .env.example to .env "
            "and paste your Google Gemini API key from https://aistudio.google.com/app/apikey"
        )

    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    # If the caller didn't specify a temperature, use the value from .env
    # (or 0.3 as a safe default -> fairly focused, not too random).
    if temperature is None:
        temperature = float(os.getenv("LLM_TEMPERATURE", "0.3"))

    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=temperature,
    )
    return llm


def get_embedding_model() -> GoogleGenerativeAIEmbeddings:
    """
    WHY: The RAG Agent (Page 5, "Ask your Dataset") needs to convert text
         into number vectors (embeddings) so it can search for similar text.
    WHAT: Returns a ready-to-use Google Gemini embedding model object.
    HOW: Reads GOOGLE_API_KEY and GEMINI_EMBEDDING_MODEL from the environment.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key == "your_google_gemini_api_key_here":
        raise ValueError(
            "GOOGLE_API_KEY is missing. Please copy .env.example to .env "
            "and paste your Google Gemini API key from https://aistudio.google.com/app/apikey"
        )

    embedding_model_name = os.getenv("GEMINI_EMBEDDING_MODEL", "models/text-embedding-004")

    embeddings = GoogleGenerativeAIEmbeddings(
        model=embedding_model_name,
        google_api_key=api_key,
    )
    return embeddings
