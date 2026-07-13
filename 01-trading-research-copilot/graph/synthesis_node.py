"""LLM node that synthesizes HTF bias + LTF candidates into a markdown research note."""

from __future__ import annotations

import os

from dotenv import find_dotenv, load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from synthesis_prompt import SYNTHESIS_SYSTEM_PROMPT, format_state_for_prompt

# find_dotenv() walks up from this file's directory looking for the nearest
# .env, so it picks up either a project-specific .env in
# 01-trading-research-copilot/ or the repo-root .env, whichever exists.
load_dotenv(find_dotenv())

DEFAULT_MODEL = "gpt-4o-mini"


def synthesize_node(state: dict) -> dict:
    """Call the synthesis LLM and write its markdown output into state['synthesis_output'].

    Reads OPENAI_API_KEY from the environment (loaded via python-dotenv from
    .env). On failure - missing API key, API error, etc. - records the error
    in state['errors'] instead of raising, so this last node can't crash a
    run that otherwise succeeded.

    Args:
        state: The graph state (CopilotState) accumulated by the upstream
            fetch/analysis nodes.

    Returns:
        A partial state update: {'synthesis_output': str} on success, or
        {'errors': [...]} with this failure appended on error.
    """
    try:
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Add it to a .env file in the project root."
            )

        llm = ChatOpenAI(model=DEFAULT_MODEL, temperature=0)

        user_message = format_state_for_prompt(
            ticker=state.get("ticker", ""),
            htf_bias=state.get("htf_bias"),
            ltf_candidates=state.get("ltf_candidates"),
            errors=state.get("errors", []),
        )

        response = llm.invoke(
            [
                SystemMessage(content=SYNTHESIS_SYSTEM_PROMPT),
                HumanMessage(content=user_message),
            ]
        )

        return {"synthesis_output": response.content}
    except Exception as exc:
        return {"errors": state.get("errors", []) + [f"synthesize_node: {exc}"]}
