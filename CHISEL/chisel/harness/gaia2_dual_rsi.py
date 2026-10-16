"""Native argument typing and complete usage aggregation helpers."""

from __future__ import annotations

USAGE_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens")


def well_typed(value, specification):
    kind = specification.get("type")
    if kind == "string":
        return isinstance(value, str)
    if kind == "integer":
        return type(value) is int
    if kind == "number":
        return type(value) in (int, float)
    if kind == "boolean":
        return type(value) is bool
    if kind == "array":
        return isinstance(value, list) and all(
            well_typed(item, specification.get("items", {})) for item in value
        )
    if kind == "object":
        return isinstance(value, dict)
    return False


def combine_usage(first, second):
    for usage in (first, second):
        if (any(type(usage.get(key)) is not int or usage[key] < 0 for key in USAGE_FIELDS)
                or usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]):
            raise ValueError("Both paid proposals require complete native usage")
    result = {key: first[key] + second[key] for key in USAGE_FIELDS}
    result["completion_duration"] = first["completion_duration"] + second["completion_duration"]
    return result
