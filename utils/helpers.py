# utils/helpers.py
# ------------------
# WHY: Small utility functions that several pages/agents need, but that
#      don't belong to any single agent. Keeping them here avoids copy-pasting
#      the same code in multiple files.
# WHAT: Timestamp formatting, safe folder creation, and simple execution-time
#       tracking used for the "Agent Execution Trace" bonus feature.
# HOW: Plain Python standard library only - no extra dependencies needed.

import os
import time
from datetime import datetime


def get_timestamp() -> str:
    """
    WHY: Reports need a human-readable timestamp so students know when
         a report was generated.
    WHAT: Returns the current date and time as a formatted string.
    HOW: Uses Python's built-in datetime module.
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def ensure_folder_exists(folder_path: str) -> None:
    """
    WHY: We save uploaded files and generated reports to disk. If the target
         folder doesn't exist yet, saving would crash with an error.
    WHAT: Creates the folder (and any missing parent folders) if needed.
    HOW: os.makedirs with exist_ok=True does nothing if the folder is already there.
    """
    os.makedirs(folder_path, exist_ok=True)


class StepTimer:
    """
    WHY: The "Agent Execution Trace" bonus feature shows students how long
         each agent took to run. Timing manually with time.time() everywhere
         is repetitive and error-prone, so we wrap it in a tiny helper class.
    WHAT: A simple stopwatch you start and stop around a block of code.
    HOW: Call .start() before running an agent, then .stop() right after -
         it returns the elapsed time in seconds, rounded to 2 decimals.
    """

    def __init__(self):
        self._start_time = None

    def start(self):
        # Record the current time when the timer starts.
        self._start_time = time.time()
        return self

    def stop(self) -> float:
        # Calculate how many seconds have passed since start() was called.
        if self._start_time is None:
            return 0.0
        elapsed_seconds = time.time() - self._start_time
        return round(elapsed_seconds, 2)


def format_agent_trace_entry(agent_name: str, duration_seconds: float, status: str = "completed") -> dict:
    """
    WHY: The UI shows a table of "which agent ran, how long it took, and its status".
         We need a consistent dictionary shape for every entry in that table.
    WHAT: Builds one row of the agent execution trace.
    HOW: Just packages the inputs into a dictionary with friendly keys.
    """
    return {
        "Agent": agent_name,
        "Duration (seconds)": duration_seconds,
        "Status": status,
        "Timestamp": get_timestamp(),
    }
