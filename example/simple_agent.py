from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from tool_anywhere_agent import create_tool_anywhere_agent

import uuid
import os
from dotenv import load_dotenv


load_dotenv()

model = ChatOpenAI(
    model="deepseek/deepseek-r1",
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
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

system_message = "You are a helpful assistant that can call tools to solve problems."


class OutputSchema(BaseModel):
    response: str = Field(..., description="Response to the user's question")
    explanation: str = Field(..., description="Explanation of the response")


parser = PydanticOutputParser(pydantic_object=OutputSchema)

graph = create_tool_anywhere_agent(
    model=model, tools=tools, custom_system_message=system_message, parser=parser
)
app = graph.compile()

thread_id = f"tool_anywhere_{uuid.uuid4()}"
config = {
    "configurable": {"thread_id": thread_id},
    "run_name": "tool_anywhere",
    "tags": ["tool_anywhere"],
}

examples = [
    "HELLO",                                                                                # No tool calls
    "What is 15 * 8 + 42?",                                                                 # Calculator tool
    "How many characters are in the word 'hello world'?",                                   # Text length tool
    "Count the words in: 'The quick brown fox jumps over the lazy dog'",                    # Word count tool
    "Calculate 100 / 4 and tell me how many words are in 'artificial intelligence'",        # Non dependent tool calls
    "Calculate 100 * 10 and then tell me how many characters are in the result number",     # Dependent tool calls
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

        parsed_result = parser.parse(response['messages'][-1].content)
        print(parsed_result.response)
        print(parsed_result.explanation)

        # for message in response["messages"]:
        #     message.pretty_print()

    except Exception as e:
        print(f"Error: {e}")
