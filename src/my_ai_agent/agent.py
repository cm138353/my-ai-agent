"""The orchestration loop: ask the model, run requested tools, repeat."""

import json

from openai import OpenAI
from openai.types.responses import (
    EasyInputMessageParam,
    Response,
    ResponseFunctionToolCall,
    ResponseInputParam,
)
from openai.types.responses.response_input_item_param import FunctionCallOutput

from my_ai_agent.tools import TOOL_FUNCTIONS, TOOL_SCHEMAS

MODEL = "gpt-5.1"
MAX_STEPS = 10  # safety limit so a confused model can't loop forever


def run_tool(name: str, arguments: str) -> str:
    """Execute one tool call and return its result as a string."""
    function = TOOL_FUNCTIONS.get(name)
    if function is None:
        return f"Error: unknown tool '{name}'"
    try:
        result = function(**json.loads(arguments))
    except Exception as exc:  # report errors to the model instead of crashing
        return f"Error: {exc}"
    return json.dumps(result)


def run_agent(client: OpenAI, user_input: str) -> str:
    """Run the agent loop until the model gives a final text answer."""
    input_items: ResponseInputParam = [
        EasyInputMessageParam(role="user", content=user_input)
    ]

    for _ in range(MAX_STEPS):
        response: Response = client.responses.create(
            model=MODEL,
            input=input_items,
            tools=TOOL_SCHEMAS,
        )

        # Keep the model's output (including tool requests) in the history.
        input_items += response.output

        tool_calls: list[ResponseFunctionToolCall] = [
            item for item in response.output if item.type == "function_call"
        ]
        if not tool_calls:
            return response.output_text

        for call in tool_calls:
            print(f"[tool] {call.name}({call.arguments})")
            input_items.append(
                FunctionCallOutput(
                    type="function_call_output",
                    call_id=call.call_id,
                    output=run_tool(call.name, call.arguments),
                )
            )

    return "Stopped: reached the maximum number of steps."
