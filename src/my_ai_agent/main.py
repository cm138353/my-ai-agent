from dotenv import load_dotenv
from openai import OpenAI

from my_ai_agent.agent import run_agent


def main() -> None:
    load_dotenv()
    client = OpenAI()
    answer = run_agent(
        client, "What time is it right now, and what is 1234.5 plus 6789.25?"
    )
    print(answer)


if __name__ == "__main__":
    main()
