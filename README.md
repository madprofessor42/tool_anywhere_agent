# Tool Anywhere Agent

A production-ready LangGraph agent that enables tool calling for any LLM, regardless of native tool support.

## Overview

Tool Anywhere Agent provides a clean, efficient way to add tool calling capabilities to any language model. Built on LangGraph's agent architecture, it integrates seamlessly into existing workflows while maintaining optimal performance.

## Key Features

**Native LangGraph Architecture**: Built as a genuine LangGraph agent with proper state management, routing, and tool execution nodes.

**Automatic Prompt Engineering**: Dynamically generates system prompts based on your tools - no manual prompt crafting required.

**Efficient Request Pattern**: Uses a direct model→route→tools flow instead of multi-step planning, reducing LLM API calls.

**Smart State Management**: Automatically tracks completed tool calls to prevent redundant operations.

**Custom System Role**: You can specify a custom system role to control the agent's behavior and persona by passing.

**Custom Output Parser**: You can define and use your own Pydantic output parser for structured responses.

**Zero Configuration**: Works out of the box with any LangChain tools and models.

## Installation

```bash
pip install tool-anywhere-agent
```

## License

MIT License
