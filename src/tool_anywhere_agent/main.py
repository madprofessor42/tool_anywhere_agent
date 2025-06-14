from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool
from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
)
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from langgraph.graph.message import add_messages
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import ToolNode

from typing import Sequence, TypedDict
from typing_extensions import Annotated

from .prompt_builder import (
    create_system_message,
    prepare_tools,
    convert_llm_response,
)


class AgentState(TypedDict):
    """The state of the agent."""
    messages: Annotated[Sequence[BaseMessage], add_messages]


def _route_tools(
    state: AgentState,
):
    messages = state["messages"]
    last_message = messages[-1]

    if last_message.tool_calls:
        return "tools"
    return END


def create_tool_anywhere_agent(
    model: BaseLanguageModel,
    tools: Sequence[BaseTool] = None,
    parser: PydanticOutputParser = None,
    custom_system_message: str = None,
):
    available_tools = prepare_tools(tools=tools)

    class OutputSchema(BaseModel):
        response: str = Field(..., description="Response to the user's question")

    is_custom_parser = parser is not None
    if parser is None:
        simple_output_parser = PydanticOutputParser(pydantic_object=OutputSchema)
    else:
        simple_output_parser = parser

    def _call_model_node(state: AgentState):
        messages = state.get("messages", [])

        # Create system message with current conversation history to check for completed tool calls
        system_message_content = create_system_message(
            tools=available_tools,
            messages=messages,
            parser=simple_output_parser,
            custom_system_message=custom_system_message,
        )
        system_msg = SystemMessage(content=system_message_content)

        response = model.invoke([system_msg] + messages)

        converted_message = convert_llm_response(
            response, parser=simple_output_parser, is_custom_parser=is_custom_parser
        )

        return {"messages": converted_message}

    _execute_tools_node = ToolNode(tools)

    graph = StateGraph(state_schema=AgentState)

    graph.add_node("model_node", _call_model_node)
    graph.add_node("tools_node", _execute_tools_node)

    graph.add_edge(START, "model_node")
    graph.add_conditional_edges(
        "model_node",
        _route_tools,
        {"tools": "tools_node", END: END},
    )
    # Any time a tool is called, we return to the chatbot to decide the next step
    graph.add_edge("tools_node", "model_node")

    return graph