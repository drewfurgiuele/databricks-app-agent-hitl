"""
Health Claims Agent — LangGraph agent with Genie MCP tools and
human-in-the-loop approval before every tool call.
"""

import logging
from databricks_langchain import (
    ChatDatabricks,
    DatabricksMCPServer,
    DatabricksMultiServerMCPClient,
)
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
from databricks.sdk import WorkspaceClient

logger = logging.getLogger(__name__)

# --------------- configuration ---------------
GENIE_SPACE_ID = "01f08a5f354519db92cd69d45c5918ff"   # Health Claims Data Room
LLM_ENDPOINT = "external_catalog.default.apim-gpt-6-sol"

SYSTEM_PROMPT = (
    "You are a helpful Health Claims assistant. You have access to the "
    "Health Claims Data Room through a Genie Agent tool. Use this tool to "
    "answer questions about health claims data.\n\n"
    "When you need to query the data room, clearly explain what you intend "
    "to ask before doing so. Be concise and professional in your responses."
)


class HealthClaimsAgent:
    """LangGraph ReAct agent with Genie MCP tools and HITL approval."""

    def __init__(self):
        self.agent = None
        self.checkpointer = MemorySaver()
        self._pending_threads: set[str] = set()
        self._mcp_client = None

    # ---------- lifecycle ----------

    async def initialize(self):
        """Connect to the Genie MCP server and build the agent graph."""
        w = WorkspaceClient()
        host = w.config.host.rstrip("/")

        genie_url = f"{host}/api/2.0/mcp/genie/{GENIE_SPACE_ID}"
        logger.info("Connecting to Genie MCP at %s", genie_url)

        self._mcp_client = DatabricksMultiServerMCPClient([
            DatabricksMCPServer(
                name="health-claims-genie",
                url=genie_url,
                workspace_client=w,
            ),
        ])

        tools = await self._mcp_client.get_tools()
        logger.info("Loaded %d MCP tool(s)", len(tools))

        llm = ChatDatabricks(endpoint=LLM_ENDPOINT)

        self.agent = create_react_agent(
            llm,
            tools=tools,
            checkpointer=self.checkpointer,
            interrupt_before=["tools"],   # ← HITL: pause before execution
            prompt=SYSTEM_PROMPT,
        )
        logger.info("Agent initialised successfully")

    # ---------- public API ----------

    def is_pending_approval(self, thread_id: str) -> bool:
        return thread_id in self._pending_threads

    async def send_message(self, message: str, thread_id: str) -> str:
        """Send a user message and return the agent's text response."""
        if self.agent is None:
            return "⚠️ Agent not yet initialised. Please refresh the page."

        config = {"configurable": {"thread_id": thread_id}}

        try:
            if self.is_pending_approval(thread_id):
                return await self._handle_approval(message, thread_id, config)

            result = await self.agent.ainvoke(
                {"messages": [HumanMessage(content=message)]},
                config=config,
            )
            return self._process_result(result, thread_id)

        except Exception as exc:
            logger.exception("Error processing message")
            self._pending_threads.discard(thread_id)
            return f"⚠️ An error occurred: {exc}"

    # ---------- internals ----------

    async def _handle_approval(
        self, message: str, thread_id: str, config: dict
    ) -> str:
        """Resume or reject a paused tool call based on user input."""
        approved = message.lower().strip() in (
            "yes", "y", "approve", "go ahead", "proceed", "ok", "sure",
        )
        self._pending_threads.discard(thread_id)

        if approved:
            result = await self.agent.ainvoke(None, config=config)
            return self._process_result(result, thread_id)

        return (
            "Understood — I won't execute that query. "
            "Is there anything else I can help with?"
        )

    def _process_result(self, result: dict, thread_id: str) -> str:
        """Extract a text answer, or surface a HITL prompt if interrupted."""
        messages = result.get("messages", [])
        if not messages:
            return "I wasn't able to generate a response."

        last_msg = messages[-1]

        # Interrupted before a tool call → ask the human
        if getattr(last_msg, "tool_calls", None):
            self._pending_threads.add(thread_id)
            return self._format_approval_request(last_msg.tool_calls)

        if getattr(last_msg, "content", None):
            return last_msg.content

        return "I wasn't able to generate a response."

    @staticmethod
    def _format_approval_request(tool_calls: list) -> str:
        lines = [
            "🔍 **Human Approval Required**\n",
            "I'd like to query the Health Claims Data Room:\n",
        ]
        for tc in tool_calls:
            lines.append(f"**Tool:** `{tc['name']}`")
            args = tc.get("args", {})
            if isinstance(args, dict):
                for k, v in args.items():
                    lines.append(f"  - **{k}:** {v}")
            else:
                lines.append(f"  - **Input:** {args}")
            lines.append("")
        lines.append("Would you like me to proceed? Reply **yes** or **no**.")
        return "\n".join(lines)
