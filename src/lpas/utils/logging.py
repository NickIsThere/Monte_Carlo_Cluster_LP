from __future__ import annotations

from typing import Any


def format_kv(**values: Any) -> str:
    """Return a lightweight key=value log string."""

    parts = [f"{key}={value}" for key, value in values.items()]
    return " ".join(parts)
