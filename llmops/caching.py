# llmops/caching.py
# --------------------
# WHY: If two users (or the same user twice) ask the EXACT same question,
#      there's no reason to pay for and wait on a second LLM call - we
#      already know the answer. Caching trades a tiny bit of disk space for
#      a HUGE speed and cost win on repeated prompts.
# WHAT: Turns on LangChain's built-in LLM response cache, backed by a local
#       SQLite file. Once enabled, EVERY agent in this project automatically
#       benefits - no changes needed anywhere else, because LangChain checks
#       the cache before making any API call.
# HOW: LangChain exposes a single global switch: set_llm_cache(...). We point
#      it at a SQLite-backed cache, which (unlike a plain in-memory cache)
#      also survives restarting the app.

import os
import warnings

from langchain_core.globals import set_llm_cache

CACHE_FILE_PATH = os.path.join("reports", "llm_cache.sqlite")

_cache_enabled = False


def enable_llm_caching() -> str:
    """
    WHY: This should run ONCE when the app starts up - after this, every
         llm.invoke(...) call anywhere in the project is automatically cached.
    WHAT: Points LangChain's global cache at a local SQLite file and returns
          the file path (useful for showing students where the cache lives).
    HOW: `langchain_community`'s SQLiteCache implements LangChain's cache
         interface (lookup + update) using a simple local database file.
    """
    global _cache_enabled

    os.makedirs("reports", exist_ok=True)

    # SQLiteCache currently lives in langchain_community, which prints a
    # gentle "this package is being sunset" warning on import - that's about
    # langchain_community as a whole, not this specific class, so we silence
    # it here to keep the teaching demo's output clean.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from langchain_community.cache import SQLiteCache

    set_llm_cache(SQLiteCache(database_path=CACHE_FILE_PATH))
    _cache_enabled = True
    return CACHE_FILE_PATH


def is_caching_enabled() -> bool:
    """WHAT: Lets the UI show a status indicator (cache ON/OFF)."""
    return _cache_enabled


def time_llm_call(llm, prompt: str) -> dict:
    """
    WHY: The best way to TEACH caching is to show the speed difference, not
         just describe it.
    WHAT: Calls the given LLM with a prompt and measures how long it took.
    HOW: A plain wall-clock timer around llm.invoke(). Call this twice with
         the SAME prompt to see a cache miss, then a near-instant cache hit.
    """
    import time

    start_time = time.time()
    response = llm.invoke(prompt)
    elapsed_seconds = round(time.time() - start_time, 3)

    return {"response_text": response.content, "elapsed_seconds": elapsed_seconds}
