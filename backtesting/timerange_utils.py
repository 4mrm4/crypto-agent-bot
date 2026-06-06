"""Timerange sanitisation utilities.

Converts various LLM-invented date formats to Freqtrade's ``YYYYMMDD-YYYYMMDD``
format used throughout the codebase.
"""

import logging
import re

logger = logging.getLogger(__name__)


def sanitize_timerange(raw: str) -> str:
    """Convert any LLM-invented date format to freqtrade's ``YYYYMMDD-YYYYMMDD``.

    Handles all common variants:
      "2024-01-01/2024-12-31" -> "20240101-20241231"
      "2024-01-01-2024-12-31" -> "20240101-20241231"
      "2024-01-01"            -> "20240101-"
      "20240101-20241231"     -> unchanged (already valid)
      "20240101-"             -> unchanged
      "2024"                  -> "2024"
    """
    raw = raw.strip()
    if not raw:
        return "20210101-"
    # Separate by / or whitespace first, then by dash
    # Extract all digit groups: "2024-01-01/2024-12-31" -> [2024,01,01,2024,12,31]
    groups = re.findall(r'\d+', raw)
    if not groups:
        return "20210101-"
    # If there's a "/" or "-" separator between dates, groups are split into two dates
    # Detect: if groups have 6+ entries, treat as two dates of 3 groups each (YYYY MM DD)
    if len(groups) >= 6:
        # Two dates: first 3 groups = date1, next 3 = date2
        d1 = "".join(groups[:3])[:8]
        d2 = "".join(groups[3:6])[:8]
        return f"{d1}-{d2}"
    if len(groups) == 5:
        # Two dates, first has 3 groups, second has 2 (YYYY MM -> YYYYMM)
        d1 = "".join(groups[:3])[:8]
        d2 = "".join(groups[3:])[:8]
        return f"{d1}-{d2}"
    if len(groups) == 4:
        # Could be YYYYMMDD-YYYYMMDD split, or YYYY MM DD YYYY
        # If any group has length 2, it's likely YYYY MM DD YYYY
        if any(len(g) <= 2 for g in groups):
            d1 = "".join(groups[:2])[:8]
            d2 = "".join(groups[2:])[:8]
            return f"{d1}-{d2}"
        # Otherwise it's already two 8-digit dates
        return f"{groups[0][:8]}-{groups[1][:8]}"
    if len(groups) == 3:
        # Single date in YYYY MM DD format
        d = "".join(groups)[:8]
        return f"{d}-"
    if len(groups) == 2:
        # Could be YYYYMMDD-YYYYMMDD without hyphen, or YYYY MM alone
        if all(len(g) == 8 for g in groups):
            return f"{groups[0][:8]}-{groups[1][:8]}"
        # Two groups: likely YYYY and MM -> pad
        d = "".join(groups)[:8]
        return f"{d}-" if len(d) == 8 else d
    # Single group: "20240101" or "2024" or "20240101-"
    d = groups[0][:8]
    if len(d) == 8 and raw.endswith('-'):
        return f"{d}-"
    return f"{d}-" if len(d) == 8 else d
