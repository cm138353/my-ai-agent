from dotenv import load_dotenv
from openai import OpenAI, OpenAIError
from openai.types.responses import (
    EasyInputMessageParam,
    ResponseInputParam,
)

from my_ai_agent.agent import run_agent


def main() -> None:
    load_dotenv()
    client = OpenAI()

    input_items: ResponseInputParam = []
    try:
        while True:
            user_input: str = input("You: ")
            if user_input.strip().lower() in ("", "quit", "exit"):
                break
            # Remember the history size so a failed turn can be rolled back.
            checkpoint = len(input_items)
            input_items += [EasyInputMessageParam(role="user", content=user_input)]
            try:
                answer = run_agent(client, input_items)
            except OpenAIError as exc:
                # Drop everything added this turn so no tool request is left
                # without its result.
                del input_items[checkpoint:]
                print(f"[error] {exc}")
                continue
            print(answer)
    except KeyboardInterrupt, EOFError:
        print("\nGoodbye.")


if __name__ == "__main__":
    main()
