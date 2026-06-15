from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd


FEEDBACK_COLUMNS = [
    "timestamp",
    "query",
    "restaurant_name",
    "feedback",
    "reason",
]


def log_feedback(
    query: str,
    restaurant_name: str,
    feedback: str,
    reason: str = "",
    path: str | Path = "reports/feedback.csv",
) -> dict:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "query": query,
        "restaurant_name": restaurant_name,
        "feedback": feedback,
        "reason": reason,
    }
    df = pd.DataFrame([row], columns=FEEDBACK_COLUMNS)
    header = not path.exists()
    df.to_csv(path, mode="a", header=header, index=False, encoding="utf-8-sig")
    return row


def load_feedback(path: str | Path = "reports/feedback.csv") -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=FEEDBACK_COLUMNS)
    return pd.read_csv(path)
