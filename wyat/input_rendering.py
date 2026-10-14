"""Explicit, label-free input rendering for an opted-in native artifact.

Legacy artifacts do not select a policy. The API-Bank policy moves complete
catalog schemas to candidates. The v3 policy checks schemas already inline.
Neither policy supplies new tool information.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path
from typing import TypedDict

from wyat import tasks

API_BANK_SCHEMA_NEXT = "api-bank-schema-next-v1"
TOOL_DECISION_SCHEMA_INLINE_V3 = "tool-decision-schema-inline-v3"
TOOL_DECISION_SCHEMA_INLINE_V4_1 = "tool-decision-schema-inline-v4.1"
TOOL_DECISION_SCHEMA_INLINE_V5 = "tool-decision-schema-inline-v5"
API_BANK_TASK = "API-Bank current API tool selection"
TOOL_DECISION_V3_TASK = "Tool decision with clarification and no tool v3"
TOOL_DECISION_V4_1_TASK = "Tool decision with clarification and no tool v4.1"
TOOL_DECISION_V5_TASK = "Tool decision with clarification and no tool v5"
TOOL_DECISION_V3_INSTRUCTIONS = (
    "Choose the next justified action from the complete tool schemas, ask for "
    "a missing required user field, or state that no listed tool supports the "
    "request. Select one action only. Do not invent arguments or execute a tool."
)
TOOL_DECISION_V3_CLARIFY = (
    "Ask the user for the required field before selecting or invoking a tool; "
    "do not invent a value."
)
TOOL_DECISION_V3_NO_TOOL = (
    "No listed tool performs the requested operation on the requested resource."
)
API_BANK_INSTRUCTIONS = (
    "Select the single API tool to invoke for the dialogue prefix in state. "
    "Use the published catalog in state. Do not infer arguments or use "
    "the current/future API result."
)
API_BANK_NEUTRAL_INSTRUCTIONS = API_BANK_INSTRUCTIONS.replace(
    "Use the published catalog in state.",
    "Use the published catalog provided with this question.",
)


class InputRenderingPolicy(TypedDict):
    """The selected rendering behavior and its executable source identity."""

    mode: str
    source_sha256: str


def input_rendering_policy(mode: str | None = None) -> InputRenderingPolicy | None:
    """Return a fresh authenticated opt-in policy; omission means legacy."""
    if mode is None:
        return None
    if type(mode) is not str or mode not in (
        API_BANK_SCHEMA_NEXT,
        TOOL_DECISION_SCHEMA_INLINE_V3,
        TOOL_DECISION_SCHEMA_INLINE_V4_1,
        TOOL_DECISION_SCHEMA_INLINE_V5,
    ):
        raise ValueError("unsupported input rendering mode")
    return {
        "mode": mode,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def validate_input_rendering_policy(policy: object) -> InputRenderingPolicy | None:
    """Reject unknown modes, extra fields, and changed renderer source."""
    if policy is None:
        return None
    if type(policy) is not dict or set(policy) != {"mode", "source_sha256"}:
        raise ValueError("input rendering policy requires mode and source_sha256")
    mode = policy["mode"]
    if type(mode) is not str:
        raise ValueError("unsupported input rendering mode")
    expected = input_rendering_policy(mode)
    if policy != expected:
        raise ValueError("input rendering source identity differs")
    return expected


def render_record(record: object, policy: object = None) -> object:
    """Apply the selected schema policy to a label-free public record.

    The task markers keep each policy narrow. Recognized but malformed inputs
    fail rather than dropping schemas or guessing a tool binding. The API-Bank
    mode expects an original catalog; v3 expects complete inline schemas.
    """
    selected = validate_input_rendering_policy(policy)
    if selected is None:
        return record
    # Enforce the existing label-free public boundary before copying any input.
    tasks.public_questions(record)
    assert isinstance(record, dict)
    if selected["mode"] == TOOL_DECISION_SCHEMA_INLINE_V3:
        is_complete_v3_tool_decision(record)
        return record
    if selected["mode"] == TOOL_DECISION_SCHEMA_INLINE_V4_1:
        is_complete_v4_1_tool_decision(record)
        return record
    if selected["mode"] == TOOL_DECISION_SCHEMA_INLINE_V5:
        is_complete_v5_tool_decision(record)
        return record
    state = record["state"]
    if type(state) is not dict or state.get("task") != API_BANK_TASK:
        return record
    questions = record["questions"]
    if set(questions) != {"tool"} or questions["tool"]["type"] != "choice":
        raise ValueError("API-Bank rendering requires one tool Choice question")
    question = questions["tool"]
    if question.get("instructions") not in (
        API_BANK_INSTRUCTIONS,
        API_BANK_NEUTRAL_INSTRUCTIONS,
    ):
        raise ValueError("unsupported API-Bank tool instructions")
    dialogue = state.get("dialogue")
    if type(dialogue) is not str and not (
        type(dialogue) is list and all(type(turn) is dict for turn in dialogue)
    ):
        raise ValueError("API-Bank rendering requires dialogue text or turn objects")
    catalog = state.get("available_tools")
    if type(catalog) is not list or not catalog:
        raise ValueError(
            "API-Bank rendering requires a complete available_tools catalog"
        )
    by_name = {}
    for schema in catalog:
        if type(schema) is not dict or set(schema) != {
            "name",
            "description",
            "input_parameters",
            "output_parameters",
        }:
            raise ValueError("API-Bank tool schema fields differ")
        name = schema["name"]
        if type(name) is not str or not name.strip() or name in by_name:
            raise ValueError("API-Bank tool names must be nonempty and unique")
        if type(schema["description"]) is not str or not schema["description"].strip():
            raise ValueError("API-Bank tool descriptions must be nonempty text")
        if any(
            type(schema[key]) is not dict
            for key in ("input_parameters", "output_parameters")
        ):
            raise ValueError("API-Bank parameter schemas must be objects")
        by_name[name] = schema
    criteria = question["criteria"]
    names = list(criteria.values())
    if len(set(names)) != len(names) or set(names) != set(by_name):
        raise ValueError("API-Bank candidates must bind the complete catalog once")
    rendered = copy.deepcopy(record)
    del rendered["state"]["available_tools"]
    rendered_question = rendered["questions"]["tool"]
    rendered_question["instructions"] = API_BANK_NEUTRAL_INSTRUCTIONS
    rendered_question["criteria"] = {
        key: json.dumps(
            by_name[name],
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        for key, name in criteria.items()
    }
    return rendered


def _v3_field_schema(field: object) -> bool:
    if type(field) is not dict or type(field.get("description")) is not str:
        return False
    if not field["description"].strip():
        return False
    kind = field.get("type")
    if type(kind) is not str or kind not in {
        "string",
        "number",
        "integer",
        "boolean",
        "array",
    }:
        return False
    allowed = {"type", "description"}
    if kind == "string":
        allowed |= {"format", "enum"}
        if "format" in field and (
            type(field["format"]) is not str
            or field["format"] not in {"email", "date-time"}
        ):
            return False
        if "enum" in field and (
            type(field["enum"]) is not list
            or not field["enum"]
            or any(type(value) is not str or not value for value in field["enum"])
            or len(set(field["enum"])) != len(field["enum"])
        ):
            return False
    elif kind == "integer":
        allowed.add("minimum")
        if "minimum" in field and type(field["minimum"]) is not int:
            return False
    elif kind == "array":
        allowed |= {"items", "minItems"}
        if (
            field.get("items") != {"type": "string"}
            or type(field.get("minItems")) is not int
            or field["minItems"] < 1
        ):
            return False
    return set(field) <= allowed


def _v3_tool_schema(schema: object) -> bool:
    if type(schema) is not dict or set(schema) != {
        "name",
        "description",
        "input_parameters",
        "output_parameters",
    }:
        return False
    if any(
        type(schema[key]) is not str or not schema[key].strip()
        for key in ("name", "description")
    ):
        return False
    inputs = schema["input_parameters"]
    outputs = schema["output_parameters"]
    if (
        type(inputs) is not dict
        or set(inputs) != {"type", "properties", "required", "additionalProperties"}
        or inputs["type"] != "object"
        or inputs["additionalProperties"] is not False
        or type(inputs["properties"]) is not dict
        or type(inputs["required"]) is not list
    ):
        return False
    properties = inputs["properties"]
    required = inputs["required"]
    if (
        len(properties) < 2
        or len(required) != len(properties) - 1
        or any(type(name) is not str or not name.strip() for name in properties)
        or any(type(name) is not str for name in required)
        or len(set(required)) != len(required)
        or not set(required) <= set(properties)
        or any(not _v3_field_schema(field) for field in properties.values())
    ):
        return False
    return outputs == {
        "type": "object",
        "properties": {
            "result": {"type": "string", "description": "Outcome or retrieved record."}
        },
        "required": ["result"],
    }


def is_complete_v3_tool_decision(record: object) -> bool:
    """Validate the already inline, label-free v3 schema menu exactly once."""
    tasks.public_questions(record)
    assert isinstance(record, dict)
    state = record["state"]
    if type(state) is not dict or state.get("task") != TOOL_DECISION_V3_TASK:
        return False
    if (
        set(state) != {"task", "dialogue"}
        or type(state["dialogue"]) is not str
        or not state["dialogue"].strip()
    ):
        raise ValueError("v3 tool decision requires a dialogue only")
    questions = record["questions"]
    if set(questions) != {"tool"}:
        raise ValueError("v3 tool decision requires one tool question")
    question = questions["tool"]
    if (
        set(question) != {"type", "instructions", "criteria"}
        or question["type"] != "choice"
        or question["instructions"] != TOOL_DECISION_V3_INSTRUCTIONS
    ):
        raise ValueError("v3 tool decision question differs")
    criteria = question["criteria"]
    if (
        type(criteria) is not dict
        or len(criteria) not in {8, 14, 27, 55}
        or list(criteria) != [f"option_{index}" for index in range(len(criteria))]
    ):
        raise ValueError("v3 tool decision menu size or order differs")
    names: set[str] = set()
    meta_options: set[str] = set()
    for value in criteria.values():
        if type(value) is not str:
            raise ValueError("v3 tool option must be text")
        if value in {TOOL_DECISION_V3_CLARIFY, TOOL_DECISION_V3_NO_TOOL}:
            if value in meta_options:
                raise ValueError("v3 tool decision repeats an action")
            meta_options.add(value)
            continue
        try:
            schema = json.loads(value)
            canonical = json.dumps(
                schema,
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (TypeError, ValueError) as error:
            raise ValueError("v3 tool option is not canonical JSON") from error
        if value != canonical or not _v3_tool_schema(schema):
            raise ValueError("v3 tool option schema differs")
        if schema["name"] in names:
            raise ValueError("v3 tool names must be unique")
        names.add(schema["name"])
    if meta_options != {TOOL_DECISION_V3_CLARIFY, TOOL_DECISION_V3_NO_TOOL}:
        raise ValueError("v3 tool decision requires both meta actions")
    return True


def _v4_1_field_schema(field: object) -> bool:
    """Validate the bounded JSON Schema subset used by authored v4.1 tools."""
    if type(field) is not dict:
        return False
    kind = field.get("type")
    if type(kind) is not str or kind not in {
        "string",
        "integer",
        "number",
        "boolean",
        "array",
        "object",
    }:
        return False
    if "description" in field and (
        type(field["description"]) is not str or not field["description"].strip()
    ):
        return False
    allowed = {"type", "description"}
    if kind == "string":
        allowed |= {"minLength", "maxLength", "pattern", "format", "enum"}
        for key in ("minLength", "maxLength"):
            if key in field and (type(field[key]) is not int or field[key] < 0):
                return False
        if field.get("minLength", 0) > field.get("maxLength", float("inf")):
            return False
        if "format" in field and (
            type(field["format"]) is not str
            or field["format"] not in {"date", "date-time", "uri", "email"}
        ):
            return False
        if "pattern" in field:
            if type(field["pattern"]) is not str:
                return False
            try:
                re.compile(field["pattern"])
            except re.error:
                return False
        if "enum" in field and (
            type(field["enum"]) is not list
            or not field["enum"]
            or any(type(value) is not str for value in field["enum"])
            or len(set(field["enum"])) != len(field["enum"])
        ):
            return False
    elif kind in {"integer", "number"}:
        allowed |= {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"}
        bounds = ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum")
        expected = int if kind == "integer" else (int, float)
        if any(
            type(field[key]) not in ((int,) if kind == "integer" else expected)
            for key in bounds
            if key in field
        ):
            return False
        lower = max(
            (field[key] for key in ("minimum", "exclusiveMinimum") if key in field),
            default=None,
        )
        upper = min(
            (field[key] for key in ("maximum", "exclusiveMaximum") if key in field),
            default=None,
        )
        if lower is not None and upper is not None:
            if lower > upper:
                return False
            if lower == upper and (
                field.get("exclusiveMinimum") == lower
                or field.get("exclusiveMaximum") == upper
            ):
                return False
    elif kind == "array":
        allowed |= {"items", "minItems", "maxItems"}
        if "items" not in field or not _v4_1_field_schema(field["items"]):
            return False
        for key in ("minItems", "maxItems"):
            if key in field and (type(field[key]) is not int or field[key] < 0):
                return False
        if field.get("minItems", 0) > field.get("maxItems", float("inf")):
            return False
    elif kind == "object":
        allowed |= {"properties", "required", "additionalProperties"}
        properties = field.get("properties", {})
        required = field.get("required", [])
        extra = field.get("additionalProperties", False)
        if (
            type(properties) is not dict
            or type(required) is not list
            or any(type(name) is not str or not name for name in properties)
            or any(type(name) is not str for name in required)
            or len(set(required)) != len(required)
            or not set(required) <= set(properties)
            or any(not _v4_1_field_schema(value) for value in properties.values())
            or (type(extra) is not bool and not _v4_1_field_schema(extra))
            or (not properties and type(extra) is bool)
        ):
            return False
    return set(field) <= allowed


def _v4_1_tool_schema(schema: object) -> bool:
    if type(schema) is not dict or set(schema) != {
        "name",
        "description",
        "input_parameters",
        "output_parameters",
    }:
        return False
    if any(
        type(schema[key]) is not str or not schema[key].strip()
        for key in ("name", "description")
    ):
        return False
    inputs = schema["input_parameters"]
    outputs = schema["output_parameters"]
    return (
        _v4_1_field_schema(inputs)
        and _v4_1_field_schema(outputs)
        and set(inputs) == {"type", "properties", "required", "additionalProperties"}
        and inputs["type"] == "object"
        and inputs["additionalProperties"] is False
        and len(inputs["properties"]) >= 1
        and len(inputs["required"]) >= 1
        and set(outputs) == {"type", "properties", "required", "additionalProperties"}
        and outputs["type"] == "object"
        and outputs["additionalProperties"] is False
        and len(outputs["properties"]) >= 1
    )


def is_complete_v4_1_tool_decision(record: object) -> bool:
    """Validate a complete authored v4.1 menu without changing its input."""
    tasks.public_questions(record)
    assert isinstance(record, dict)
    state = record["state"]
    if type(state) is not dict or state.get("task") != TOOL_DECISION_V4_1_TASK:
        return False
    if (
        set(state) != {"task", "dialogue"}
        or type(state["dialogue"]) is not str
        or not state["dialogue"].strip()
    ):
        raise ValueError("v4.1 tool decision requires a dialogue only")
    questions = record["questions"]
    if set(questions) != {"tool"}:
        raise ValueError("v4.1 tool decision requires one tool question")
    question = questions["tool"]
    if (
        set(question) != {"type", "instructions", "criteria"}
        or question["type"] != "choice"
        or question["instructions"] != TOOL_DECISION_V3_INSTRUCTIONS
    ):
        raise ValueError("v4.1 tool decision question differs")
    criteria = question["criteria"]
    if (
        type(criteria) is not dict
        or len(criteria) != 6
        or list(criteria) != [f"option_{index}" for index in range(6)]
    ):
        raise ValueError("v4.1 tool decision menu size or order differs")
    names: set[str] = set()
    meta_options: set[str] = set()
    for value in criteria.values():
        if type(value) is not str:
            raise ValueError("v4.1 tool option must be text")
        if value in {TOOL_DECISION_V3_CLARIFY, TOOL_DECISION_V3_NO_TOOL}:
            if value in meta_options:
                raise ValueError("v4.1 tool decision repeats an action")
            meta_options.add(value)
            continue
        try:
            schema = json.loads(value)
            canonical = json.dumps(
                schema,
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (TypeError, ValueError) as error:
            raise ValueError("v4.1 tool option is not canonical JSON") from error
        if value != canonical or not _v4_1_tool_schema(schema):
            raise ValueError("v4.1 tool option schema differs")
        if schema["name"] in names:
            raise ValueError("v4.1 tool names must be unique")
        names.add(schema["name"])
    if len(names) != 4 or meta_options != {
        TOOL_DECISION_V3_CLARIFY,
        TOOL_DECISION_V3_NO_TOOL,
    }:
        raise ValueError("v4.1 tool decision requires four tools and both actions")
    return True


def is_complete_v5_tool_decision(record: object) -> bool:
    """Validate a complete inline v5 menu of 2–53 tools and both actions.

    The v5 mode does not read labels or row metadata. Its schema subset remains
    strict: unconstrained nested objects are not complete output schemas.
    """
    tasks.public_questions(record)
    assert isinstance(record, dict)
    state = record["state"]
    if type(state) is not dict or state.get("task") != TOOL_DECISION_V5_TASK:
        return False
    if (
        set(state) != {"task", "dialogue"}
        or type(state["dialogue"]) is not str
        or not state["dialogue"].strip()
    ):
        raise ValueError("v5 tool decision requires a dialogue only")
    questions = record["questions"]
    if set(questions) != {"tool"}:
        raise ValueError("v5 tool decision requires one tool question")
    question = questions["tool"]
    if (
        set(question) != {"type", "instructions", "criteria"}
        or question["type"] != "choice"
        or question["instructions"] != TOOL_DECISION_V3_INSTRUCTIONS
    ):
        raise ValueError("v5 tool decision question differs")
    criteria = question["criteria"]
    if (
        type(criteria) is not dict
        or not 4 <= len(criteria) <= 55
        or list(criteria) != [f"option_{index}" for index in range(len(criteria))]
    ):
        raise ValueError("v5 tool decision menu size or order differs")
    names: set[str] = set()
    meta_options: set[str] = set()
    for value in criteria.values():
        if type(value) is not str:
            raise ValueError("v5 tool option must be text")
        if value in {TOOL_DECISION_V3_CLARIFY, TOOL_DECISION_V3_NO_TOOL}:
            if value in meta_options:
                raise ValueError("v5 tool decision repeats an action")
            meta_options.add(value)
            continue
        try:
            schema = json.loads(value)
            canonical = json.dumps(
                schema,
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (TypeError, ValueError) as error:
            raise ValueError("v5 tool option is not canonical JSON") from error
        if value != canonical or not _v4_1_tool_schema(schema):
            raise ValueError("v5 tool option schema differs")
        if schema["name"] in names:
            raise ValueError("v5 tool names must be unique")
        names.add(schema["name"])
    if len(names) != len(criteria) - 2 or meta_options != {
        TOOL_DECISION_V3_CLARIFY,
        TOOL_DECISION_V3_NO_TOOL,
    }:
        raise ValueError("v5 tool decision requires tools and both actions")
    return True
