# Modifications Copyright (c) 2026 Tatsuhiro Shimizu

from __future__ import annotations

import datetime
from typing import Any


def ope_unix_times(synthetic: Any) -> dict[str, int]:
    """Resolve ISO timestamps from config (naive local), matching legacy conf.py."""
    return {
        "t_oldest": int(datetime.datetime.fromisoformat(synthetic.t_oldest_iso).timestamp()),
        "t_now": int(datetime.datetime.fromisoformat(synthetic.t_now_iso).timestamp()),
        "t_future": int(datetime.datetime.fromisoformat(synthetic.t_future_iso).timestamp()),
        "time_at_evaluation": int(
            datetime.datetime.fromisoformat(synthetic.time_at_evaluation_iso).timestamp()
        ),
    }


def opl_unix_times(synthetic: Any) -> dict[str, int]:
    return {
        "t_oldest": int(datetime.datetime.fromisoformat(synthetic.t_oldest_iso).timestamp()),
        "t_now": int(datetime.datetime.fromisoformat(synthetic.t_now_iso).timestamp()),
        "t_future": int(datetime.datetime.fromisoformat(synthetic.t_future_iso).timestamp()),
        "time_at_evaluation_start": int(
            datetime.datetime.fromisoformat(synthetic.time_at_evaluation_start_iso).timestamp()
        ),
    }
