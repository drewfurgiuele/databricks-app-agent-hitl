"""
Sample agent with tools and HITL
"""

import os
import re
import logging
from databricks_langchain import (
    ChatDatabricks,
    DatabricksMCPServer,
    DatabricksMultiServerMCPClient,
)
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from databricks.sdk import WorkspaceClient

logger = logging.getLogger(__name__)

# --------------- configuration ---------------
# Set in app.yaml when deployed, or in .env when running locally.

def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(
            f"Missing required environment variable {name}. "
            "Set it in app.yaml (deployed) or .env (local)."
        )
    return value


SYSTEM_PROMPT = (
    "You are a helpful Health Claims assistant. You have access to the "
    "Health Claims Data Room through a Genie Agent tool. Use this tool to "
    "answer questions about health claims data.\n\n"
    "When you need to query the data room, clearly explain what you intend "
    "to ask before doing so. If the user declines a query, do not retry it "
    "unless they ask you to. Be concise and professional in your responses."
)

DECLINED_TOOL_RESULT = "The user declined this tool call; it was not executed."

# --------------- approval parsing ---------------
# A reply to an approval prompt is only treated as yes/no when every word is
# in the vocabulary below. Anything else (e.g. "no, check 2024 instead") is a
# new instruction for the agent.

_APPROVE_WORDS = {
    "yes", "y", "yeah", "yep", "yup", "sure", "ok", "okay", "k", "approve",
    "approved", "proceed", "go", "run", "do", "lgtm", "sounds", "good",
    "fine", "absolutely", "definitely",
}
_REJECT_WORDS = {
    "no", "n", "nope", "nah", "cancel", "stop", "decline", "declined",
    "reject", "skip", "don't", "dont", "not", "never", "mind",
}
_FILLER_WORDS = {
    "please", "thanks", "thank", "you", "it", "that", "this", "ahead",
    "just", "now", "and",
}
_REPLY_VOCAB = _APPROVE_WORDS | _REJECT_WORDS | _FILLER_WORDS


def classify_reply(text: str) -> str:
    """Return "approve", "reject", or "other" for a reply to an approval prompt."""
    words = re.findall(r"[a-z']+", text.lower().replace("\u2019", "'"))
    if not words or any(w not in _REPLY_VOCAB for w in words):
        return "other"
    # negation wins, so "don't do it" is a rejection
    if any(w in _REJECT_WORDS for w in words):
        return "reject"
    if any(w in _APPROVE_WORDS for w in words):
        return "approve"
    return "other"


class HealthClaimsAgent:
    """LangGraph ReAct agent with Genie MCP tools and HITL approval."""

    def __init__(self):
        # read config eagerly so a misconfigured app fails at startup
        self.genie_space_id = _require_env("GENIE_SPACE_ID")
        self.llm_endpoint = _require_env("LLM_ENDPOINT")
        self.agent = None
        self.checkpointer = MemorySaver()
        self._pending_threads: set[str] = set()
        self._mcp_client = None

    # ---------- lifecycle ----------

    async def initialize(self):
        """Connect to the Genie MCP server and build the agent graph."""
        w = WorkspaceClient()
        host = w.config.host.rstrip("/")

        genie_url = f"{host}/api/2.0/mcp/genie/{self.genie_space_id}"
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

        # Route through AI Gateway's Responses API:
        # POST {host}/ai-gateway/mlflow/v1/responses with the UC model name
        # in the request body (not /serving-endpoints/<name>/invocations).
        llm = ChatDatabricks(
            model=self.llm_endpoint,
            workspace_client=w,
            use_ai_gateway=True,
            use_responses_api=True,
        )

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
            try:
                await self._decline_pending_tool_calls(config)
            except Exception:
                logger.exception("Failed to clean up pending tool calls")
            return f"⚠️ An error occurred: {exc}"

    # ---------- internals ----------

    async def _handle_approval(
        self, message: str, thread_id: str, config: dict
    ) -> str:
        """Resume, reject, or redirect a paused tool call based on user input."""
        decision = classify_reply(message)
        self._pending_threads.discard(thread_id)

        if decision == "approve":
            result = await self.agent.ainvoke(None, config=config)
            return self._process_result(result, thread_id)

        if decision == "reject":
            reply = (
                "Understood — I won't execute that query. "
                "Is there anything else I can help with?"
            )
            await self._decline_pending_tool_calls(config, reply=reply)
            return reply

        # Not a plain yes/no: decline the pending call and treat the message
        # as a new instruction, e.g. "no, look at 2024 instead".
        await self._decline_pending_tool_calls(config)
        result = await self.agent.ainvoke(
            {"messages": [HumanMessage(content=message)]},
            config=config,
        )
        return self._process_result(result, thread_id)

    async def _decline_pending_tool_calls(
        self, config: dict, reply: str | None = None
    ) -> None:
        """Record a "declined" result for any tool calls that never ran.

        Chat endpoints reject histories where a tool call has no matching
        tool result, so this must happen before the thread takes another
        user message.
        """
        state = await self.agent.aget_state(config)
        messages = state.values.get("messages", [])
        tool_calls = getattr(messages[-1], "tool_calls", None) if messages else None
        if not tool_calls:
            return

        updates = [
            ToolMessage(
                content=DECLINED_TOOL_RESULT,
                tool_call_id=tc["id"],
                name=tc["name"],
            )
            for tc in tool_calls
        ]
        if reply:
            updates.append(AIMessage(content=reply))

        # Writing as the "agent" node routes the graph to END, so the thread
        # is no longer parked in front of the tools node.
        await self.agent.aupdate_state(
            config, {"messages": updates}, as_node="agent"
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
        lines.append(
            "Would you like me to proceed? Reply **yes** to run it, **no** to "
            "cancel, or tell me what to change."
        )
        return "\n".join(lines)
