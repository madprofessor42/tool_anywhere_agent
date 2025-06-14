from langchain.output_parsers import PydanticOutputParser
from langchain_core.tools import BaseTool
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ToolMessage,
)

from pydantic import BaseModel, Field
from typing import Sequence, TypedDict, Any, List

import re
import json


class ToolDefinition(TypedDict):
    name: str
    description: str
    args: dict[str, Any]


def get_completed_tool_calls(messages: Sequence[BaseMessage]) -> str:
    """
    Analyze message history to find completed tool calls and their results.
    
    Args:
        messages: List of messages in the conversation
        
    Returns:
        str: Formatted string with completed tool call results
    """
    completed_calls = []
    
    # Create a mapping of tool call IDs to their results
    tool_results = {}
    
    # First pass: collect all tool results
    for message in messages:
        if isinstance(message, ToolMessage):
            tool_results[message.tool_call_id] = {
                "name": message.name if hasattr(message, 'name') else "unknown",
                "content": message.content
            }
    
    # Second pass: find AI messages with tool calls and match them with results
    for message in messages:
        if isinstance(message, AIMessage) and hasattr(message, 'tool_calls') and message.tool_calls:
            for tool_call in message.tool_calls:
                tool_call_id = tool_call.get('id')
                if tool_call_id and tool_call_id in tool_results:
                    result_info = tool_results[tool_call_id]
                    completed_calls.append({
                        'tool_name': tool_call.get('name'),
                        'args': tool_call.get('args', {}),
                        'result': result_info['content']
                    })
    
    if not completed_calls:
        return ""
    
    # Format completed tool calls for the prompt
    completed_section = "The following tools have already been called with these exact arguments. DO NOT call them again with the same arguments:\n\n"
    
    for i, call in enumerate(completed_calls, 1):
        completed_section += f"{i}. Tool: {call['tool_name']}\n"
        completed_section += f"   Arguments: {call['args']}\n"
        completed_section += f"   Result: {call['result']}\n\n"
    
    completed_section += "If you need the result of any of these previous tool calls, use the result shown above instead of calling the tool again.\n\n"
    
    return completed_section


def create_system_message(
    tools: list[ToolDefinition] = None,
    messages: Sequence[BaseMessage] = None,
    parser: PydanticOutputParser = None,
    custom_system_message: str = None,
) -> str:
    """
    Create a system message with tool instructions and JSON schema.

    Args:
        tools (ToolDefinition): List of available tools
        messages: List of conversation messages to check for completed tool calls
        parser: Output parser to use
        custom_system_message: Custom system message to use
    Returns:
        str: Formatted system message with JSON schema instructions
    """

    class ToolCall(BaseModel):
        tool: str = Field(..., description="Name of the tool to call")
        args: dict = Field(..., description="Arguments to pass to the tool")

    tool_call_parser = PydanticOutputParser(pydantic_object=ToolCall)

    # Get completed tool calls information
    completed_tools_info = ""
    if messages:
        completed_tools_info = get_completed_tool_calls(messages)

    # Формируем детальное описание каждого инструмента
    tools_description = ""

    for tool in tools:
        tools_description += f"\n## Tool Name: {tool['name']}\n"
        tools_description += f"Tool Description: {tool['description']}\n"

        if tool["args"]:
            tools_description += "Arguments:\n"
            for arg_name, arg_info in tool["args"].items():
                # Handle different type structures
                if "anyOf" in arg_info:
                    # Handle anyOf structure (like your 'b' parameter)
                    types = []
                    for type_option in arg_info["anyOf"]:
                        types.append(type_option.get("type", "unknown"))
                    arg_type = " | ".join(types)
                else:
                    # Handle simple type structure (like your 'a' parameter)
                    arg_type = arg_info.get("type", "any")

                # Determine if required (no default value and not nullable)
                has_default = "default" in arg_info
                is_nullable = "anyOf" in arg_info and any(
                    t.get("type") == "null" for t in arg_info["anyOf"]
                )
                required = (
                    "Required" if not has_default and not is_nullable else "Optional"
                )

                # Add default value info if present
                default_info = ""
                if has_default:
                    default_info = f", default: {arg_info['default']}"

                tools_description += (
                    f"- {arg_name} ({arg_type}, {required}{default_info})\n"
                )

    if custom_system_message is None:
        custom_system_message = "You are a tool calling agent. You have a list of tools and your task is to determine which tool to use."

    sys_msg = (
        f"# System Role:\n"
        f"{custom_system_message}\n\n"
        f"# List of available tools:\n"
        f"{tools_description}\n\n"
        f"# Previously completed tool calls:\n"
        f"{completed_tools_info}"
        f"# Tool calling rules:\n"
        f"1. When a user's question matches a tool's capability, you MUST use that tool. "
        f"2. Do not try to solve problems manually if a tool exists for that purpose. "
        f"3. If tool calls do not depend on each other, you can call them in a sequence. "
        f"4. NEVER call the same tool with the same arguments if it has already been completed (see above). "
        f"5. If the user's question doesn't require any tool, answer directly and do not invoke a tool at all. "
        f"6. System role specified in the beginning is a must to follow. "
        f"7. Do not mention to user anything about tool calling, just answer the question.\n\n"
        f"If there were tool calls, output ONLY a JSON object (with no extra text) that adheres EXACTLY to the following schema:\n\n"
        f"{tool_call_parser.get_format_instructions()}\n\n"
        f"If there were no tool calls, output ONLY a JSON object (with no extra text) that adheres EXACTLY to the following schema:\n\n"
        f"{parser.get_format_instructions()}\n\n"
    )

    return sys_msg


def prepare_tools(tools: Sequence[BaseTool]) -> list[ToolDefinition]:
    tools_definitions = [
        {
            "name": tool.name,
            "description": tool.description,
            "args": tool.args,
        }
        for tool in tools
    ]

    return tools_definitions


def convert_llm_response(response: BaseMessage, parser: PydanticOutputParser = None, is_custom_parser: bool = False) -> List[BaseMessage]:
    """
    Convert LLM response to appropriate LangChain messages.
    If response contains tool calls (JSON), convert to AIMessage with tool_calls.
    Otherwise, return as AIMessage.
    
    Args:
        response: The LLM response message
        parser: The output parser to use
        is_custom_parser: Whether the parser is a custom parser (True) or default parser (False)
    """
    converted_messages = []

    # If it's already a BaseMessage, extract content
    if isinstance(response, BaseMessage):
        content = response.content
    else:
        content = str(response)

    json_matches = re.findall(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", content, re.DOTALL)

    tool_calls = []

    if json_matches:
        for match in json_matches:
            try:
                data = json.loads(match)

                if data.get("tool"):
                    # Try to parse as tool calls
                    tool_name = data.get("tool")
                    tool_args = data.get("args", {})

                    if tool_name:
                        # Create tool call structure
                        tool_call = {
                            "name": tool_name,
                            "args": tool_args,
                            "id": f"{tool_name}_{hash(str(tool_args))}",
                        }
                        tool_calls.append(tool_call)
                else:
                    # When no tool is being called, use the custom schema if it's passed and do not parse the response
                    if is_custom_parser:
                        content = match
                    # Otherwise, use the default parser and parse the response
                    else:
                        try:
                            parsed_response = parser.parse(match)
                            content = parsed_response.response
                        except Exception as e:
                            # If parsing fails, keep the original content
                            pass
                    
            except json.JSONDecodeError:
                # Skip invalid JSON
                continue

    if tool_calls:
        # Create AIMessage with tool calls
        ai_message = AIMessage(content=content, tool_calls=tool_calls)
        converted_messages.append(ai_message)
    else:
        # No valid tool calls found, treat as regular AI message
        ai_message = AIMessage(content=content)
        converted_messages.append(ai_message)

    return converted_messages
