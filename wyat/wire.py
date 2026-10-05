"""Strict JSON decoding shared by the CLI and HTTP boundary."""

import json
import math
from pathlib import Path
from typing import cast

type JsonValue = (
    None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]
)
type JsonObject = dict[str, JsonValue]


def _object(pairs: list[tuple[str, JsonValue]]) -> JsonObject:
    result: JsonObject = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON number: {value}")


def _float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("JSON number exceeds finite float range")
    return parsed


def decode_json(data: str | bytes) -> JsonValue:
    # json.loads is dynamically typed; the decoder only creates JSON values.
    return cast(
        JsonValue,
        json.loads(
            data,
            object_pairs_hook=_object,
            parse_constant=_constant,
            parse_float=_float,
        ),
    )


def read_object(path: Path) -> JsonObject:
    value = decode_json(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path.name}")
    return value
