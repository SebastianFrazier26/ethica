"""Classification of stored episode errors."""

from __future__ import annotations

import re

_OVERFLOW_RE = re.compile(
    r"context (?:length|window|size)|maximum context|too long|"
    r"exceed\w*[^.]{0,60}(?:context|token)|token limit|sequence length|max_position",
    re.IGNORECASE,
)


def is_context_overflow(error: str | None) -> bool:
    """Deterministic failure: the same prompt overflows again, so retrying cannot help.
    Matched on message text because backends raise their own exception types."""
    return bool(error and _OVERFLOW_RE.search(error))
