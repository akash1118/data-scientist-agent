# llmops/evaluation.py
# -----------------------
# WHY: An LLM will confidently answer even when it's wrong. "Evaluation" is
#      how you systematically check whether an AI answer is actually good,
#      instead of just trusting it because it sounds fluent.
# WHAT: Two beginner-friendly evaluation techniques:
#         1. Groundedness check (no LLM call, instant, free) - does the
#            answer actually use words that appear in the retrieved context,
#            or does it look "made up"?
#         2. LLM-as-judge (one extra LLM call) - ask a second LLM prompt to
#            score the answer against the question on a simple 1-5 rubric.
# HOW: Groundedness uses plain string/set operations (word overlap). The
#      LLM judge reuses our existing get_llm() - same pattern as every other
#      agent in this project.

import re


def _extract_significant_words(text: str) -> set:
    """
    WHY: Comparing whole sentences is noisy - comparing the MEANINGFUL words
         (ignoring tiny common words like "the", "is", "a") is a much better
         signal for "does this answer overlap with the source data?"
    WHAT: Returns a lowercase set of words with length > 3 from the text.
    HOW: A simple regex split - no NLP library needed for a teaching example.
    """
    words = re.findall(r"[a-zA-Z]+", text.lower())
    return {word for word in words if len(word) > 3}


def evaluate_groundedness(answer: str, context_chunks: list) -> dict:
    """
    WHY: This is a fast, free, "good enough for a demo" way to catch obvious
         hallucination - an answer that shares almost no words with the
         retrieved context is suspicious.
    WHAT: Returns a groundedness score (0-100%) and a verdict label.
    HOW: Computes what fraction of the answer's significant words also
         appear somewhere in the retrieved context.
    """
    answer_words = _extract_significant_words(answer)
    context_text = " ".join(context_chunks)
    context_words = _extract_significant_words(context_text)

    if not answer_words:
        return {"score_percent": 0.0, "verdict": "Empty answer - nothing to evaluate."}

    overlapping_words = answer_words & context_words
    score_percent = round((len(overlapping_words) / len(answer_words)) * 100, 1)

    if score_percent >= 60:
        verdict = "Well grounded - most of the answer's key words appear in the retrieved data."
    elif score_percent >= 30:
        verdict = "Partially grounded - some words trace back to the data, some don't."
    else:
        verdict = "Weakly grounded - the answer barely overlaps with the retrieved data. Possible hallucination."

    return {"score_percent": score_percent, "verdict": verdict}


LLM_JUDGE_PROMPT_TEMPLATE = """
You are grading an AI assistant's answer for quality. Be strict but fair.

Question: {question}
Answer: {answer}

Rate the answer from 1 (bad) to 5 (excellent) on how well it addresses the
question. Reply with ONLY this exact format, nothing else:
Score: <number>
Reason: <one short sentence>
"""


def evaluate_with_llm_judge(question: str, answer: str) -> dict:
    """
    WHY: This demonstrates "LLM-as-a-judge" - a very common real-world
         evaluation pattern where a second LLM call grades the first one's
         output, instead of (or in addition to) a human reviewing it.
    WHAT: Returns {"score": int, "reason": str}.
    HOW: Sends a small grading prompt to our existing get_llm(), then parses
         the "Score: N" / "Reason: ..." lines out of the reply.
    """
    from utils.llm import get_llm

    prompt = LLM_JUDGE_PROMPT_TEMPLATE.format(question=question, answer=answer)
    llm = get_llm()
    response = llm.invoke(prompt)
    reply_text = response.content

    score_match = re.search(r"Score:\s*(\d)", reply_text)
    reason_match = re.search(r"Reason:\s*(.+)", reply_text)

    return {
        "score": int(score_match.group(1)) if score_match else None,
        "reason": reason_match.group(1).strip() if reason_match else reply_text.strip(),
    }
