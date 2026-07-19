"""Append-only JSONL logging for hyperparameter searches."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Mapping


def append_search_record(log_dir: str, search_name: str, record: Mapping[str, Any]) -> str:
    """Append one timestamped trial without ever overwriting an earlier run."""
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, f"{search_name}.jsonl")
    payload = {"timestamp": datetime.now(timezone.utc).isoformat(), **dict(record)}
    with open(path, "a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    return path
