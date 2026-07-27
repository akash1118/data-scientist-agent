# llmops/monitoring.py
# -----------------------
# WHY: Once an AI app is running for real users, you need to know: is it
#      working? How slow is it? How often does it fail? "Monitoring" means
#      systematically recording that information instead of only finding out
#      when a user complains.
# WHAT: A simple event logger. Every agent run gets logged with a timestamp,
#       duration, and success/failure status - both in memory (for the live
#       Streamlit dashboard) and appended to a CSV file (so it survives
#       restarts, like a real monitoring system would).
# HOW: One class, MonitoringLog, with a single `log_event()` method. No
#      external monitoring service needed - this is a from-scratch, minimal
#      version of what tools like Datadog or Prometheus do at a larger scale.

import os
import csv
from datetime import datetime

LOG_FILE_PATH = os.path.join("reports", "monitoring_log.csv")
LOG_FIELDS = ["timestamp", "agent", "duration_seconds", "status", "detail"]


class MonitoringLog:
    """
    WHY: Keeping the log in a small class (instead of loose functions) makes
         it easy to hold an in-memory copy for a fast Streamlit dashboard,
         while still persisting every event to disk.
    WHAT: log_event() records one event; get_events() and get_stats() read
          them back for display.
    """

    def __init__(self):
        self.events = []
        os.makedirs("reports", exist_ok=True)
        self._ensure_csv_header()

    def _ensure_csv_header(self):
        """WHY: A CSV file needs its header written exactly once."""
        if not os.path.exists(LOG_FILE_PATH):
            with open(LOG_FILE_PATH, "w", newline="", encoding="utf-8") as csv_file:
                csv.DictWriter(csv_file, fieldnames=LOG_FIELDS).writeheader()

    def log_event(self, agent_name: str, duration_seconds: float, status: str = "success", detail: str = "") -> dict:
        """
        WHY: Called by app.py right after every agent run, success or failure.
        WHAT: Records one monitoring event, in memory AND on disk.
        HOW: Appends a dict to self.events, then appends the same row to the CSV.
        """
        event = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "agent": agent_name,
            "duration_seconds": duration_seconds,
            "status": status,
            "detail": detail,
        }
        self.events.append(event)

        with open(LOG_FILE_PATH, "a", newline="", encoding="utf-8") as csv_file:
            csv.DictWriter(csv_file, fieldnames=LOG_FIELDS).writerow(event)

        return event

    def get_events(self) -> list:
        """WHAT: Returns every event logged so far (most recent last)."""
        return self.events

    def get_stats(self) -> dict:
        """
        WHY: Raw event lists are hard to eyeball - a monitoring dashboard
             needs quick summary numbers.
        WHAT: Returns call count, error count/rate, and average latency.
        """
        if not self.events:
            return {"total_calls": 0, "error_count": 0, "error_rate_percent": 0.0, "avg_duration_seconds": 0.0}

        total_calls = len(self.events)
        error_count = sum(1 for e in self.events if e["status"] != "success")
        avg_duration = sum(e["duration_seconds"] for e in self.events) / total_calls

        return {
            "total_calls": total_calls,
            "error_count": error_count,
            "error_rate_percent": round((error_count / total_calls) * 100, 1),
            "avg_duration_seconds": round(avg_duration, 2),
        }
