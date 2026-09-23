"""Explicit external conversation state; neither weights nor a KV cache.

Only save(path) and load(path) access a caller-supplied local file. The store
does not infer facts, grant consent, or promote assistant guesses into records.
"""
from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class Message:
    role: str
    content: str

    def __post_init__(self):
        if self.role not in {"user", "assistant"}:
            raise ValueError("History roles are user or assistant in this example")
        if not isinstance(self.content, str) or not self.content.strip():
            raise ValueError("Message content must be nonempty text")


def format_history(messages) -> str:
    """A visible serialization, not a model-specific chat template."""
    messages = tuple(messages)
    if any(not isinstance(message, Message) for message in messages):
        raise ValueError("Expected Message records")
    return json.dumps([asdict(message) for message in messages],
                      ensure_ascii=False, separators=(",", ":"))


@dataclass(frozen=True)
class Summary:
    text: str
    source_turns: tuple[int, ...]
    omitted_turns: tuple[int, ...]


def summarize_history(messages, *, max_chars: int) -> Summary:
    """Keep the first nonempty line of each turn if the whole line fits.

    This intentionally limited extractive baseline is deterministic. It can
    omit qualifications on later lines. source_turns records provenance, not
    a guarantee that all information in those turns was preserved.
    max_chars counts Python code points, NOT model tokens.
    """
    if type(max_chars) is not int or max_chars < 0:
        raise ValueError("max_chars must be a nonnegative integer")
    messages = tuple(messages)
    format_history(messages)  # Validate without mutating anything.
    lines, included, omitted = [], [], []
    for number, message in enumerate(messages, 1):
        first = next(line.strip() for line in message.content.splitlines() if line.strip())
        line = f"{number}:{message.role}: {first}"
        if len("\n".join(lines+[line])) <= max_chars:
            lines.append(line)
            included.append(number)
        else:
            omitted.append(number)
    return Summary("\n".join(lines), tuple(included), tuple(omitted))


@dataclass(frozen=True)
class MemoryRecord:
    key: str
    value: str
    source: str

    def __post_init__(self):
        if not all(isinstance(value, str) and value.strip()
                   for value in (self.key, self.value, self.source)):
            raise ValueError("Memory records require nonempty key, value and source")


class MemoryStore:
    """A key-based store with explicit replacement/deletion and JSON persistence."""
    def __init__(self, records=()):
        self._records = {}
        for record in records:
            if not isinstance(record, MemoryRecord) or record.key in self._records:
                raise ValueError("Initial memory records need unique keys")
            self.upsert(record)

    def upsert(self, record: MemoryRecord):
        if not isinstance(record, MemoryRecord):
            raise ValueError("Expected a MemoryRecord")
        self._records[record.key] = record

    def forget(self, key: str) -> bool:
        return self._records.pop(key, None) is not None

    def records(self) -> tuple[MemoryRecord, ...]:
        return tuple(self._records.values())

    def save(self, path):
        """Explicitly write/replace exactly the caller-selected JSON file."""
        Path(path).write_text(json.dumps({"version": 1, "records": [
            asdict(record) for record in self.records()
        ]}, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if (not isinstance(value, dict) or set(value) != {"version", "records"}
                or type(value["version"]) is not int or value["version"] != 1
                or not isinstance(value["records"], list)):
            raise ValueError("Invalid memory format")
        records = []
        for row in value["records"]:
            if not isinstance(row, dict) or set(row) != {"key", "value", "source"}:
                raise ValueError("Invalid memory record")
            records.append(MemoryRecord(**row))
        return cls(records)
