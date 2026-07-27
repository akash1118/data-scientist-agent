# llmops/guardrails.py
# -----------------------
# WHY: An AI app that blindly sends any user input to an LLM, and blindly
#      shows any LLM output to a user, is fragile: someone could type a huge
#      wall of text, try to manipulate the system prompt, or the LLM could
#      return something empty/broken. "Guardrails" are simple checks placed
#      BEFORE the LLM call (input) and AFTER it (output) to catch problems.
# WHAT: Two functions - validate_user_input() and validate_llm_output() -
#       each returning (is_allowed: bool, reason: str).
# HOW: Plain string checks and regex. No extra libraries needed - real
#      guardrail systems (like Guardrails AI or NeMo Guardrails) do the same
#      basic idea, just with many more, more sophisticated rules.

import re

MAX_INPUT_LENGTH = 500

# A tiny illustrative blocklist of phrases that suggest someone is trying to
# manipulate the AI ("prompt injection") rather than ask a genuine question
# about the dataset. Real systems use much larger, smarter lists than this.
SUSPICIOUS_INPUT_PATTERNS = [
    "ignore previous instructions",
    "ignore all previous instructions",
    "disregard the above",
    "you are now",
    "reveal your system prompt",
    "reveal your instructions",
]

# A pattern that looks like it might be a credit-card-shaped number, used as
# a simple, illustrative example of an OUTPUT guardrail that would catch
# obviously-sensitive-looking data before it's shown to a user. It only
# matches a single CONTIGUOUS long digit run, or the classic "4111-1111-
# 1111-1111" grouped format - NOT ordinary sentences that happen to mention
# several separate numbers (e.g. a list of salaries), which would otherwise
# trigger constant false alarms in a data-analysis app.
CREDIT_CARD_LOOKING_PATTERN = re.compile(r"\b\d{13,19}\b|\b(?:\d{4}[- ]){3}\d{4}\b")


def validate_user_input(text: str) -> tuple:
    """
    WHY: Runs BEFORE we send anything to the LLM - catches obviously bad
         input early and cheaply (no API call wasted).
    WHAT: Returns (is_allowed, reason).
    HOW: A short chain of simple checks: empty?, too long?, looks like a
         prompt-injection attempt?
    """
    if not text or not text.strip():
        return False, "Input is empty."

    if len(text) > MAX_INPUT_LENGTH:
        return False, f"Input is too long ({len(text)} characters). Please keep it under {MAX_INPUT_LENGTH}."

    lowered_text = text.lower()
    for pattern in SUSPICIOUS_INPUT_PATTERNS:
        if pattern in lowered_text:
            return False, f"Input looks like a prompt-injection attempt (matched: '{pattern}')."

    return True, "Input looks safe."


def validate_llm_output(text: str) -> tuple:
    """
    WHY: Runs AFTER the LLM responds - catches broken or risky-looking
         output before it reaches the user.
    WHAT: Returns (is_allowed, reason).
    HOW: Checks for an empty response and for text that LOOKS like it might
         contain sensitive data (a simple pattern-matching example, not a
         real PII detector).
    """
    if not text or not text.strip():
        return False, "The AI returned an empty response."

    if CREDIT_CARD_LOOKING_PATTERN.search(text):
        return False, "Output contains a number that looks like it could be sensitive (e.g. a card number). Blocked."

    return True, "Output looks safe."
