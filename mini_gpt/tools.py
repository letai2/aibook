"""A deliberately tiny, offline tool boundary: data is never executable code."""

from dataclasses import dataclass
import json


MAX_ARGUMENT = 1_000_000
OPERATIONS = ("add", "subtract", "multiply")
TOOL_SCHEMA = {
    "type": "object",
    "required": ["name", "arguments"],
    "additionalProperties": False,
    "properties": {
        "name": {"enum": list(OPERATIONS)},
        "arguments": {
            "type": "object", "required": ["a", "b"], "additionalProperties": False,
            "properties": {key: {"type": "integer", "minimum": -MAX_ARGUMENT,
                                  "maximum": MAX_ARGUMENT} for key in ("a", "b")},
        },
    },
}


class ToolError(ValueError):
    """An invalid request, not permission to execute a fallback command."""


def strict_object(text):
    """Parse one bounded JSON object; duplicate keys and nonfinite numbers fail."""
    if not isinstance(text, str) or not 1 <= len(text) <= 4096:
        raise ToolError("expected 1..4096 characters of JSON")

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ToolError("duplicate JSON key")
            result[key] = value
        return result

    def invalid_constant(_):
        raise ToolError("nonfinite JSON number")

    try:
        result = json.loads(text, object_pairs_hook=pairs, parse_constant=invalid_constant)
    except (json.JSONDecodeError, RecursionError) as error:
        raise ToolError("invalid JSON object") from error
    if type(result) is not dict:
        raise ToolError("expected a JSON object")
    return result


@dataclass(frozen=True)
class ToolCall:
    name: str
    a: int
    b: int

    def payload(self):
        return {"name": self.name, "arguments": {"a": self.a, "b": self.b}}


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    name: str = ""
    value: int | None = None
    error: str = ""


def validate_call(payload):
    if isinstance(payload, str):
        payload = strict_object(payload)
    if type(payload) is not dict or set(payload) != {"name", "arguments"}:
        raise ToolError("a call needs exactly name and arguments")
    name, arguments = payload["name"], payload["arguments"]
    if type(name) is not str or name not in OPERATIONS:
        raise ToolError("tool is not allowlisted")
    if type(arguments) is not dict or set(arguments) != {"a", "b"}:
        raise ToolError("arguments need exactly a and b")
    for value in arguments.values():
        # bool is an int subclass in Python, but is not a valid duration here.
        if type(value) is not int or abs(value) > MAX_ARGUMENT:
            raise ToolError("arguments must be bounded integers, not bool or strings")
    return ToolCall(name, arguments["a"], arguments["b"])


def execute_tool(payload):
    """Return a structured observation, including validation failures."""
    try:
        call = validate_call(payload)
    except ToolError as error:
        return ToolResult(False, error=str(error))
    if call.name == "add":
        value = call.a + call.b
    elif call.name == "subtract":
        value = call.a - call.b
    else:
        value = call.a * call.b
    return ToolResult(True, call.name, value)
