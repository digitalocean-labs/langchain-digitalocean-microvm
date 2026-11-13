"""The mock mail server's MCP tools as LangChain tools.

The course loads them with langchain-mcp-adapters, which needs `mcp<2` while the current
deepagents-code needs `mcp>=2`; so this bridges them with FastMCP's client instead: one
StructuredTool per tool the server lists, each call a short-lived client session.
"""

from __future__ import annotations

import json

from fastmcp import Client
from langchain_core.tools import StructuredTool


async def load_mail_tools(url: str) -> list[StructuredTool]:
    """Discover the server's tools at `url`; raises if the server is not reachable."""
    async with Client(url) as client:
        listed = await client.list_tools()

    # A factory (not an inline lambda/loop body) so each tool closes over its own name.
    def bridge(name: str, description: str, schema: dict) -> StructuredTool:
        async def call(**arguments) -> str:
            async with Client(url) as client:
                result = await client.call_tool(name, arguments)
            # FastMCP wraps non-object returns (lists) as {"result": ...}; unwrap that for the model.
            if result.structured_content is not None:
                return json.dumps(result.structured_content.get("result", result.structured_content))
            return "\n".join(getattr(block, "text", "") for block in result.content)

        return StructuredTool(name=name, description=description, args_schema=schema, coroutine=call)

    return [bridge(tool.name, tool.description or tool.name, tool.input_schema) for tool in listed]
