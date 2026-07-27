# agents/insight_agent.py
# --------------------------
# WHY: Raw numbers (from the Profiler Agent) don't tell a story on their own.
#      The Insight Agent uses an LLM to turn those numbers into a plain-English
#      business narrative - the "AI" part of "AI Data Scientist".
# WHAT: The second node in our LangGraph workflow. Takes profiling_result
#       from the state, sends it to Gemini, and stores the LLM's written
#       insights back into the state.
# HOW: Builds a prompt describing the profiling JSON, calls our shared
#      get_llm() from utils/llm.py, and saves the response text.

from graph.state import AgentState
from utils.llm import get_llm
from llmops.token_manager import extract_token_usage


# A single, clear prompt template. Keeping it as a plain f-string (instead of
# a complex prompt-templating library) keeps this beginner-friendly.
INSIGHT_PROMPT_TEMPLATE = """
You are a friendly, expert data analyst. A student has uploaded a dataset and
you have been given its statistical profile below (in JSON-like format).

Dataset profile:
{profiling_result}

Using ONLY the information above, write a clear, beginner-friendly analysis
with these exact sections (use markdown headings):

## Executive Summary
A short 2-3 sentence overview of the dataset.

## Business Insights
3-5 bullet points of useful business observations.

## Important Trends
2-3 bullet points describing patterns you notice in the numbers.

## Outliers
Point out any values that look unusually high, low, or suspicious.

## Interesting Relationships
Mention any notable correlations between numeric columns, if present.

## Recommendations
2-3 practical, actionable recommendations based on the data.

## Future Improvements
2-3 suggestions for what additional data or analysis would help.

Keep the language simple, avoid jargon, and keep the whole answer under 400 words.
"""


def run_insight_agent(state: AgentState) -> AgentState:
    """
    WHY: This is the LangGraph node that turns "raw stats" into "AI insights".
    WHAT: Reads profiling_result from state, asks Gemini to analyze it, and
          writes the generated text into state["ai_insights"].
    HOW: Fill in the prompt template with the profiling JSON, call the LLM,
         and store the response.
    """
    profiling_result = state.get("profiling_result", {})

    # Fill in the prompt template with the actual profiling data.
    prompt = INSIGHT_PROMPT_TEMPLATE.format(profiling_result=profiling_result)

    # Get our shared Gemini chat model and ask it to generate insights.
    llm = get_llm()
    response = llm.invoke(prompt)

    # ChatGoogleGenerativeAI responses have a `.content` attribute with the text.
    state["ai_insights"] = response.content

    # LLMOps: record how many tokens this call used (see llmops/token_manager.py).
    state["token_usage"] = extract_token_usage(response)

    # Log this step for the execution trace.
    messages = state.get("messages", [])
    messages.append({"role": "agent", "content": "Insight Agent: AI insights generated using Gemini."})
    state["messages"] = messages

    return state
