"""Native provider usage validation and literal response normalization."""

from __future__ import annotations
import re

USAGE_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens")


def native_usage(raw, model, context):
    usage = raw.get("usage", {})
    if (raw.get("model") != model
            or any(type(usage.get(key)) is not int or usage[key] < 0 for key in USAGE_FIELDS)
            or usage["prompt_tokens"] + usage["completion_tokens"] != usage["total_tokens"]
            or not 0 < usage["total_tokens"] <= context):
        raise ValueError("Incomplete or inconsistent native provider usage")
    return {key: usage[key] for key in USAGE_FIELDS}


def native_response_text(text, stop_sequences, *, judge=False):
    if not isinstance(text, str):
        raise RuntimeError("Native model response lacks text")
    # Quoted tool arguments and case-sensitive judge verdicts are literal data.
    if not judge:
        text = re.sub(
            r'"(?:\\.|[^"\\])*"|\b(?:False|True)\b',
            lambda match: match[0] if match[0].startswith('"') else match[0].lower(),
            text,
            flags=re.DOTALL,
        )
    for stop in stop_sequences:
        text = text.split(stop)[0]
    return text
