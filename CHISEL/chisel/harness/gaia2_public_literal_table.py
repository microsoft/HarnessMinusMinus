"""Reversible tables for repeated public record schemas; never evaluates tool text."""

from __future__ import annotations
import ast
from functools import lru_cache
import json
import math
import re


class UnsupportedLiteral(ValueError):
    pass


def encode_literal(text):
    parsed = ast.parse(text, mode="eval")
    if sum(1 for _ in ast.walk(parsed)) > 200000:
        raise UnsupportedLiteral("public_literal_too_large")
    schemas = []
    indices = {}

    def pack(node):
        if isinstance(node, ast.Constant):
            value = node.value
            if (type(value) not in (str, int, float, bool, type(None))
                    or isinstance(value, float) and not math.isfinite(value)):
                raise UnsupportedLiteral("unsupported_literal_type")
            return value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            if not isinstance(node.operand, ast.Constant) or type(node.operand.value) not in (int, float):
                raise UnsupportedLiteral("unsupported_signed_literal")
            return ["u", "-" if isinstance(node.op, ast.USub) else "+", pack(node.operand)]
        for kind, tag in ((ast.List, "l"), (ast.Tuple, "t"), (ast.Set, "s")):
            if isinstance(node, kind):
                return [tag, [pack(value) for value in node.elts]]
        if isinstance(node, ast.Dict):
            if any(key is None for key in node.keys):
                raise UnsupportedLiteral("dictionary_unpacking_forbidden")
            return ["d", [[pack(key), pack(value)] for key, value in zip(node.keys, node.values)]]
        if isinstance(node, ast.Call):
            if (not isinstance(node.func, ast.Name) or not re.fullmatch(r"[A-Z][A-Za-z0-9_]*", node.func.id)
                    or node.args or any(keyword.arg is None for keyword in node.keywords)):
                raise UnsupportedLiteral("only_named_public_record_fields_are_supported")
            fields = tuple(keyword.arg for keyword in node.keywords)
            if len(fields) != len(set(fields)):
                raise UnsupportedLiteral("duplicate_record_fields")
            key = node.func.id, fields
            if key not in indices:
                indices[key] = len(schemas)
                schemas.append([node.func.id, list(fields)])
            return ["@", indices[key], [pack(keyword.value) for keyword in node.keywords]]
        raise UnsupportedLiteral("expression_is_not_public_literal_data")

    value = pack(parsed.body)
    payload = {"format": "public-literal-table-v1", "schemas": schemas, "value": value}
    if ast.dump(decode_literal(payload)) != ast.dump(parsed.body):
        raise ValueError("Public literal schema/value round trip changed the native data")
    return payload


def decode_literal(payload, records=None):
    schemas = payload["schemas"]
    records = {} if records is None else records

    def unpack(value):
        if not isinstance(value, list):
            return ast.Constant(value=value)
        tag = value[0]
        if tag == "=":
            return records[value[1]]
        if tag == "@id":
            record = unpack(["@", value[2], value[3]])
            if value[1] in records:
                raise ValueError("Public record IDs are immutable")
            records[value[1]] = record
            return record
        if tag == "@":
            name, fields = schemas[value[1]]
            if len(fields) != len(value[2]):
                raise ValueError("Record schema and values have different lengths")
            return ast.Call(func=ast.Name(id=name, ctx=ast.Load()), args=[],
                            keywords=[ast.keyword(arg=name, value=unpack(item))
                                      for name, item in zip(fields, value[2])])
        if tag == "u":
            return ast.UnaryOp(op=ast.USub() if value[1] == "-" else ast.UAdd(),
                               operand=unpack(value[2]))
        if tag == "l":
            return ast.List(elts=[unpack(item) for item in value[1]], ctx=ast.Load())
        if tag == "t":
            return ast.Tuple(elts=[unpack(item) for item in value[1]], ctx=ast.Load())
        if tag == "s":
            return ast.Set(elts=[unpack(item) for item in value[1]])
        if tag == "d":
            return ast.Dict(keys=[unpack(item[0]) for item in value[1]],
                            values=[unpack(item[1]) for item in value[1]])
        raise ValueError("Unknown public literal table tag")

    return unpack(payload["value"])
