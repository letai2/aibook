"""Small, inspectable arithmetic experiments, not claims about hidden thinking."""

from collections import Counter
from dataclasses import dataclass

from .tools import MAX_ARGUMENT, ToolError, execute_tool


@dataclass(frozen=True)
class StepResult:
    id: str
    name: str
    a: int
    b: int
    value: int


def execute_plan(steps, *, max_steps=8):
    """Execute a bounded DAG written in sequence; strings reference earlier IDs."""
    if type(max_steps) is not int or not 1 <= max_steps <= 32:
        raise ToolError("max_steps must be in 1..32")
    if not isinstance(steps, (list, tuple)) or not 1 <= len(steps) <= max_steps:
        raise ToolError("plan must contain 1..max_steps steps")
    values, trace = {}, []
    for step in steps:
        if type(step) is not dict or set(step) != {"id", "name", "arguments"}:
            raise ToolError("step needs id, name and arguments")
        identifier = step["id"]
        if (type(identifier) is not str or not identifier.isidentifier()
                or len(identifier) > 40 or identifier in values):
            raise ToolError("step IDs must be short, unique identifiers")
        arguments = step["arguments"]
        if type(arguments) is not dict or set(arguments) != {"a", "b"}:
            raise ToolError("step arguments need a and b")
        resolved = {}
        for key, value in arguments.items():
            if isinstance(value, str):
                if value not in values:
                    raise ToolError("reference must name an earlier step")
                value = values[value]
            resolved[key] = value
        result = execute_tool({"name": step["name"], "arguments": resolved})
        if not result.ok:
            raise ToolError(result.error)
        values[identifier] = result.value
        trace.append(StepResult(identifier, result.name, resolved["a"], resolved["b"], result.value))
    return tuple(trace)


def study_minutes(sessions, days):
    """Independent task specification; it does not trust a proposed plan."""
    if not isinstance(sessions, (list, tuple)) or not sessions:
        raise ValueError("at least one session is required")
    if any(type(n) is not int or not 0 <= n <= MAX_ARGUMENT for n in sessions):
        raise ValueError("sessions must be nonnegative bounded integer minutes")
    if type(days) is not int or not 0 <= days <= 365:
        raise ValueError("days must be an integer in 0..365")
    return sum(sessions) * days


def verify_study_answer(sessions, days, answer):
    """A correct execution of the wrong plan still fails this task-level check."""
    return type(answer) is int and answer == study_minutes(sessions, days)


@dataclass(frozen=True)
class Vote:
    answer: int | None
    count: int
    total: int
    tied: bool


def majority_answer(answers):
    """Plurality with explicit abstention on ties; agreement is not truth."""
    if not isinstance(answers, (list, tuple)) or any(type(n) is not int for n in answers):
        raise ValueError("candidate answers must be a list/tuple of integers")
    if not answers:
        return Vote(None, 0, 0, False)
    counts = Counter(answers)
    largest = max(counts.values())
    winners = [answer for answer, count in counts.items() if count == largest]
    return Vote(winners[0] if len(winners) == 1 else None, largest, len(answers), len(winners) > 1)


def select_verified(answers, verifier):
    """Return one distinct accepted answer, or None if absent/ambiguous."""
    if not isinstance(answers, (list, tuple)) or any(type(n) is not int for n in answers):
        raise ValueError("candidate answers must be integers")
    accepted = {answer for answer in answers if verifier(answer)}
    return next(iter(accepted)) if len(accepted) == 1 else None
