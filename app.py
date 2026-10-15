"""
Health Claims Agent — Gradio chat UI for the LangGraph HITL agent.
Databricks App entry-point (see app.yaml).
"""

import os
import uuid
import logging
import gradio as gr
from agent import HealthClaimsAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# --------------- agent singleton ---------------
agent = HealthClaimsAgent()
_initialized = False


async def _ensure_initialized():
    """Lazy-init the agent on the first request (inside Gradio's loop)."""
    global _initialized
    if not _initialized:
        logger.info("Initialising agent…")
        await agent.initialize()
        _initialized = True
        logger.info("Agent ready")


# --------------- event handlers ---------------

async def respond(
    message: str, history: list, thread_id: str
):
    """Process a user message and return the updated chat history."""
    await _ensure_initialized()

    if not message.strip():
        return "", history, thread_id

    history = history + [[message, None]]
    response = await agent.send_message(message, thread_id)
    history[-1][1] = response
    return "", history, thread_id


def new_chat():
    """Reset the conversation with a fresh LangGraph thread."""
    return str(uuid.uuid4()), []


# --------------- Gradio layout ---------------

with gr.Blocks(
    title="Health Claims Agent",
    theme=gr.themes.Soft(),
    css="""
        #header  { text-align: center; margin-bottom: 0.5rem; }
        #chatbot { min-height: 480px; }
        footer   { display: none !important; }
    """,
) as demo:

    # per-session thread id (Gradio State is per browser tab)
    thread_id = gr.State(value=lambda: str(uuid.uuid4()))

    gr.Markdown(
        "# 🏥 Health Claims Agent\n"
        "Ask questions about health claims data. "
        "Tool calls require your approval before execution.",
        elem_id="header",
    )

    chatbot = gr.Chatbot(
        elem_id="chatbot",
        show_label=False,
        bubble_full_width=False,
    )

    with gr.Row():
        msg = gr.Textbox(
            placeholder="Ask a question about health claims…",
            show_label=False,
            scale=9,
            container=False,
        )
        send_btn = gr.Button(
            "Send", variant="primary", scale=1, min_width=80
        )

    clear_btn = gr.Button("🔄 New Chat", size="sm")

    # wire up events
    msg.submit(
        respond, [msg, chatbot, thread_id], [msg, chatbot, thread_id]
    )
    send_btn.click(
        respond, [msg, chatbot, thread_id], [msg, chatbot, thread_id]
    )
    clear_btn.click(new_chat, outputs=[thread_id, chatbot])


if __name__ == "__main__":
    port = int(os.environ.get("DATABRICKS_APP_PORT", 8000))
    demo.launch(server_name="0.0.0.0", server_port=port)
