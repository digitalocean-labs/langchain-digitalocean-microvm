"""Build the AsyncSubAgent spec a supervisor uses to reach a subagent through the gateway.

The supervisor never learns about MicroVMs: every spec points at the gateway
URL and names the graph as the gateway knows it (`graphs=` key of
SubAgentGateway). Auth rides in `Authorization: Bearer <gateway token>`
because `x-api-key` is a reserved header in the LangGraph SDK (passing it in
`headers` raises). The SDK will ALSO send `x-api-key` from
LANGGRAPH_API_KEY / LANGSMITH_API_KEY if either is set; set
LANGGRAPH_API_KEY to the gateway token so that header carries the right
value instead of the supervisor's LangSmith key.
"""

from __future__ import annotations

from deepagents import AsyncSubAgent


def async_subagent_spec(
    *,
    name: str,
    description: str,
    graph_id: str,
    gateway_url: str,
    gateway_token: str,
) -> AsyncSubAgent:
    """An AsyncSubAgent that delegates `graph_id` to the gateway at `gateway_url`."""
    return AsyncSubAgent(
        name=name,
        description=description,
        graph_id=graph_id,
        url=gateway_url.rstrip("/"),
        headers={"Authorization": f"Bearer {gateway_token}"},
    )
