from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

from tool_anywhere_agent import create_tool_anywhere_agent

import uuid
import os
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(
    model="deepseek/deepseek-r1",
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1"
)

@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression safely.

    Args:
        expression: A mathematical expression to evaluate (e.g., "2 + 2", "10 * 5")

    Returns:
        The result of the calculation or an error message.
    """
    try:
        # Basic safety check - only allow simple mathematical operations
        allowed_chars = set("0123456789+-*/(). ")
        if not all(c in allowed_chars for c in expression):
            return "Error: Expression contains invalid characters"

        result = eval(expression)
        return f"The result is: {result}"
    except Exception as e:
        return f"Error calculating '{expression}': {e}"

@tool
def text_length(text: str) -> str:
    """Count the number of characters in a text string.

    Args:
        text: The text to count characters for

    Returns:
        The character count.
    """
    return f"The text '{text}' has {len(text)} characters."

@tool
def word_count(text: str) -> str:
    """Count the number of words in a text string.

    Args:
        text: The text to count words for

    Returns:
        The word count.
    """
    words = text.split()
    return f"The text has {len(words)} words."

tools = [calculator, text_length, word_count]

graph = create_tool_anywhere_agent(model=model, tools=tools)
app = graph.compile()

thread_id = f"tool_anywhere_{uuid.uuid4()}"
config = {
    "configurable": {"thread_id": thread_id},
    "run_name": "tool_anywhere",
    "tags": ["tool_anywhere"],
}

examples = [
    "HELLO",
    "What is 15 * 8 + 42?",
    "How many characters are in the word 'hello world'?",
    "Count the words in: 'The quick brown fox jumps over the lazy dog'",
    "Calculate 100 / 4 and tell me how many words are in 'artificial intelligence'",
]

print("\n" + "=" * 60)
print("TOOL ANYWHERE AGENT - EXAMPLE INTERACTIONS")
print("=" * 60)

for i, query in enumerate(examples, 1):
    print(f"\n--- Example {i} ---")
    print(f"User: {query}")

    try:
        response = app.invoke(
            {"messages": [{"role": "user", "content": query}]}, config
        )

        for message in response["messages"]:
            message.pretty_print()

    except Exception as e:
        print(f"Error: {e}")
