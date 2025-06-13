"""
Multi-Agent Workflow using LangGraph with tool_anywhere_agent as one of the nodes.

This example demonstrates a coordinated workflow where:
1. Router Agent: Decides which specialized agent should handle the task
2. Tool Anywhere Agent: Handles tool-calling tasks (calculator, text processing, etc.)
3. Research Agent: Handles research and analysis tasks
4. Writer Agent: Handles content creation and writing tasks
5. Supervisor Agent: Coordinates the workflow and ensures task completion
"""

import os
import uuid
from typing import TypedDict, Literal, Sequence
from typing_extensions import Annotated

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from tool_anywhere_agent import create_tool_anywhere_agent
from dotenv import load_dotenv

load_dotenv()

# Initialize the language model
model = ChatOpenAI(
    model="deepseek/deepseek-r1",
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
    temperature=0.7,
)

# Define tools for the tool_anywhere_agent
@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression safely."""
    try:
        allowed_chars = set("0123456789+-*/(). ")
        if not all(c in allowed_chars for c in expression):
            return "Error: Expression contains invalid characters"
        result = eval(expression)
        return f"The result is: {result}"
    except Exception as e:
        return f"Error calculating '{expression}': {e}"


@tool
def text_analyzer(text: str) -> str:
    """Analyze text for character count, word count, and basic statistics."""
    words = text.split()
    sentences = text.split(".")
    return f"Text Analysis:\n- Characters: {len(text)}\n- Words: {len(words)}\n- Sentences: {len(sentences)}"


@tool
def data_processor(data: str) -> str:
    """Process and format data for analysis."""
    lines = data.strip().split("\n")
    processed = f"Processed {len(lines)} lines of data:\n"
    for i, line in enumerate(lines[:5], 1):  # Show first 5 lines
        processed += f"{i}. {line.strip()}\n"
    if len(lines) > 5:
        processed += f"... and {len(lines) - 5} more lines"
    return processed


# Multi-Agent State
class MultiAgentState(TypedDict):
    """The state shared across all agents in the workflow."""

    messages: Annotated[Sequence[BaseMessage], add_messages]
    user_query: str

    current_agent: str
    task_completed: bool
    final_response: str
    routing_decision: str


# Router Agent - Decides which agent should handle the task
def router_agent(state: MultiAgentState) -> MultiAgentState:
    """Route tasks to appropriate specialized agents based on content analysis."""
    messages = state.get("messages", [])
    last_message = messages[-1]
    user_query = state.get("user_query", None)

    if user_query is None:
        user_query = last_message.content.lower()

    # Routing logic based on keywords and task type
    if any(
        keyword in user_query
        for keyword in [
            "calculate",
            "count",
        ]
    ):
        routing_decision = "tool_agent"
    elif any(
        keyword in user_query
        for keyword in ["research", "find", "investigate", "analyze", "study"]
    ):
        routing_decision = "research_agent"
    elif any(
        keyword in user_query
        for keyword in ["write", "create", "compose", "draft", "story", "article"]
    ):
        routing_decision = "writer_agent"
    else:
        routing_decision = "tool_agent"  # Default to tool agent

    router_response = f"🤖 Router: Analyzing task... Routing to {routing_decision.replace('_', ' ').title()}"

    return {
        "messages": [AIMessage(content=router_response)],
        "user_query": user_query,
        "current_agent": "router",
        "routing_decision": routing_decision,
        "task_completed": False,
        "final_response": "",
    }


# Research Agent - Handles research and analysis tasks
def research_agent(state: MultiAgentState) -> MultiAgentState:
    """Handle research and analysis tasks."""
    user_query = state.get("user_query")

    research_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a Research Agent specializing in analysis and investigation.
        Your role is to provide comprehensive research-based responses.
        Focus on gathering information, analyzing patterns, and providing insights.
        Be thorough but concise in your analysis.""",
            ),
            ("human", "{query}"),
        ]
    )

    response = model.invoke(research_prompt.format_messages(query=user_query))
    research_response = f"🔍 Research Agent: {response.content}"

    return {
        "messages": [AIMessage(content=research_response)],
        "current_agent": "research",
        "task_completed": True,
        "final_response": research_response,
    }


# Writer Agent - Handles content creation tasks
def writer_agent(state: MultiAgentState) -> MultiAgentState:
    """Handle content creation and writing tasks."""
    user_query = state.get("user_query")

    writer_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a Writer Agent specializing in content creation.
        Your role is to create engaging, well-structured written content.
        Focus on clarity, creativity, and proper formatting.
        Adapt your writing style to match the requested format.""",
            ),
            ("human", "{query}"),
        ]
    )

    response = model.invoke(writer_prompt.format_messages(query=user_query))
    writer_response = f"✍️ Writer Agent: {response.content}"

    return {
        "messages": [AIMessage(content=writer_response)],
        "current_agent": "writer",
        "task_completed": True,
        "final_response": writer_response,
    }


# Supervisor Agent - Coordinates and ensures task completion
def supervisor_agent(state: MultiAgentState) -> MultiAgentState:
    """Supervise the workflow and provide final coordination."""
    messages = state.get("messages", [])
    final_response = state.get("final_response", "")
    current_agent = state.get("current_agent", "")

    # Extract the agent response for summary
    agent_response = ""
    if messages:
        last_message = messages[-1]
        if hasattr(last_message, "content"):
            agent_response = last_message.content

    supervisor_response = final_response
    # Create a comprehensive summary
    if current_agent == "tool":
        supervisor_response += f"\n\n👨‍💼 Supervisor: Task completed by Tool Agent. The requested calculations and tool operations have been executed successfully."
    elif current_agent == "research":
        supervisor_response += f"\n\n👨‍💼 Supervisor: Research completed by Research Agent. Comprehensive analysis and findings have been provided."
    elif current_agent == "writer":
        supervisor_response += f"\n\n👨‍💼 Supervisor: Content creation completed by Writer Agent. The requested written content has been generated."
    else:
        supervisor_response += f"\n\n👨‍💼 Supervisor: Task completed successfully. All workflow steps have been executed."

    # Add a summary of what was accomplished
    if agent_response:
        supervisor_response += f"\n\n📋 Summary: {agent_response.split(':', 1)[-1].strip() if ':' in agent_response else agent_response}"

    return {
        "messages": [AIMessage(content=supervisor_response)],
        "current_agent": "supervisor",
        "task_completed": True,
        "final_response": supervisor_response,
    }


# Create the tool_anywhere_agent
tools = [calculator, text_analyzer, data_processor]
tool_agent_graph = create_tool_anywhere_agent(model=model, tools=tools)
tool_agent_compiled = tool_agent_graph.compile()


def tool_anywhere_agent_node(state: MultiAgentState) -> MultiAgentState:
    """Execute the tool_anywhere_agent and return results."""
    user_query = state.get("user_query")

    tool_response = tool_agent_compiled.invoke(
        {"messages": [HumanMessage(content=user_query)]}
    )

    # Extract the final response from tool agent
    last_tool_message = tool_response["messages"][-1]
    tool_result = f"🛠️ Tool Agent: {last_tool_message.content}"

    return {
        "messages": [AIMessage(content=tool_result)],
        "current_agent": "tool",
        "task_completed": True,
        "final_response": tool_result,
    }


# Routing function
def route_to_agent(
    state: MultiAgentState,
) -> Literal["tool_agent", "research_agent", "writer_agent"]:
    """Route to the appropriate agent based on routing decision."""
    routing_decision = state.get("routing_decision", "tool_agent")
    return routing_decision


# Conditional routing for completion
def should_continue(state: MultiAgentState):
    """Determine if the workflow should continue or end."""
    if state.get("task_completed"):
        return "supervisor"
    return END


# Create the multi-agent workflow
def create_multi_agent_workflow():
    """Create and return the compiled multi-agent workflow."""

    # Create the state graph
    workflow = StateGraph(MultiAgentState)

    # Add nodes
    workflow.add_node("router", router_agent)
    workflow.add_node("tool_agent", tool_anywhere_agent_node)
    workflow.add_node("research_agent", research_agent)
    workflow.add_node("writer_agent", writer_agent)
    workflow.add_node("supervisor", supervisor_agent)

    # Add edges
    workflow.add_edge(START, "router")

    # Conditional routing from router to specialized agents
    workflow.add_conditional_edges(
        "router",
        route_to_agent,
        {
            "tool_agent": "tool_agent",
            "research_agent": "research_agent",
            "writer_agent": "writer_agent",
        },
    )

    # All agents route to supervisor for final coordination
    workflow.add_conditional_edges(
        "tool_agent", should_continue, {"supervisor": "supervisor", END: END}
    )

    workflow.add_conditional_edges(
        "research_agent", should_continue, {"supervisor": "supervisor", END: END}
    )

    workflow.add_conditional_edges(
        "writer_agent", should_continue, {"supervisor": "supervisor", END: END}
    )

    # Supervisor always ends the workflow
    workflow.add_edge("supervisor", END)

    return workflow.compile()


if __name__ == "__main__":
    """Run example interactions with the multi-agent workflow."""
    app = create_multi_agent_workflow()

    examples = [
        "Calculate 5 * 5",
        "Research the benefits of artificial intelligence in healthcare",
        "Write a short story about a robot learning to cook",
    ]

    print("\n" + "=" * 80)
    print("MULTI-AGENT WORKFLOW - EXAMPLE INTERACTIONS")
    print("=" * 80)

    for i, query in enumerate(examples, 1):
        print(f"\n--- Example {i} ---")
        print(f"User: {query}")
        print("-" * 40)

        try:
            thread_id = f"multi_agent_{uuid.uuid4()}"
            config = {
                "configurable": {"thread_id": thread_id},
                "run_name": "multi_agent_workflow",
                "tags": ["multi_agent"],
            }

            messages = app.invoke({"messages": [HumanMessage(content=query)]}, config)

            for message in messages["messages"]:
                message.pretty_print()

        except Exception as e:
            print(f"Error: {e}")

        print("-" * 40)
