"""Python functions the agent can call, plus their JSON schemas for the model."""

from collections.abc import Callable
from datetime import datetime
from typing import Any

from openai.types.responses import FunctionToolParam


def get_current_time() -> str:
    """Return the local date and time as an ISO 8601 string."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def add_numbers(a: float, b: float) -> float:
    """Return the sum of two numbers."""
    return a + b


# Maps the tool name the model uses to the real Python function.
TOOL_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "get_current_time": get_current_time,
    "add_numbers": add_numbers,
}

# Descriptions sent to the model (Responses API function-tool format).
TOOL_SCHEMAS: list[FunctionToolParam] = [
    {
        "type": "function",
        "name": "get_current_time",
        "description": "Get the current local date and time.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "add_numbers",
        "description": "Add two numbers together.",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "The first number."},
                "b": {"type": "number", "description": "The second number."},
            },
            "required": ["a", "b"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]
