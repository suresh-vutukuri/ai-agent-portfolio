"""Captures which specialist handled a query and logs the handoff to a JSONL file.

Hooks into CrewAI's tool dispatch via the crewai_event_bus, listening for
ToolUsageStartedEvent. This is deliberately NOT done via Crew(step_callback=...):
when the underlying LLM supports native function calling (the default for
OpenAI models), CrewAI's agent executor only invokes step_callback once, for
the agent's final AgentFinish - it never fires step_callback for intermediate
tool calls in that code path (see crewai/agents/crew_agent_executor.py,
_invoke_loop_native_tools). ToolUsageStartedEvent, by contrast, is emitted
from the actual tool-dispatch layer in both the native and legacy ReAct
tool-calling loops, so it reliably captures every tool call either way.

Two signals are captured per run:
  - delegations: who the manager explicitly handed the query to, read from
    the "coworker" argument of its "Delegate work to coworker" /
    "Ask question to coworker" tool calls. This is the authoritative
    "who was this routed to" signal, present even if that specialist goes
    on to call no domain tool at all (e.g. it just asks for a missing order
    ID instead of looking anything up).
  - tool_calls: which of the specialists' own mock tools actually fired.
    Used as a fallback when a delegation target couldn't be captured.
"""

from __future__ import annotations

import json
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, List, Optional

from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.tool_usage_events import ToolUsageStartedEvent

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = PROJECT_ROOT / "logs" / "handoff_log.jsonl"

# CrewAI's own (normalized) names for its built-in delegation tools - see
# crewai.utilities.agent_utils.DELEGATION_TOOL_NAMES.
DELEGATION_TOOL_NAMES = {"delegate_work_to_coworker", "ask_question_to_coworker"}

# Maps each mock tool's registered @tool name (normalized - see
# _normalize_tool_name) to the specialist that owns it.
TOOL_TO_SPECIALIST: dict[str, str] = {
    "check_invoice_status": "billing",
    "get_payment_history": "billing",
    "check_system_status": "tech",
    "lookup_error_code": "tech",
    "lookup_order": "returns",
    "check_refund_eligibility": "returns",
}

# Maps each specialist agent's role (normalized - see _normalize_role) to
# its short label. Must match the `role:` values in config/agents.yaml.
ROLE_TO_SPECIALIST: dict[str, str] = {
    "billing specialist": "billing",
    "technical support specialist": "tech",
    "returns specialist": "returns",
}


def _normalize_tool_name(name: str) -> str:
    """Normalize a tool name for matching against TOOL_TO_SPECIALIST.

    CrewAI's native function-calling loop (the default for OpenAI models)
    sanitizes tool names to lowercase snake_case for LLM-provider
    compatibility - e.g. "Check Invoice Status" becomes "check_invoice_status"
    - while the legacy ReAct loop reports the tool's original display name
    unchanged. Normalizing both forms the same way lets one dict match
    either code path.

    Args:
        name: A tool name as reported by ToolUsageStartedEvent.tool_name.

    Returns:
        The name lowercased with whitespace collapsed to single underscores.
    """
    return "_".join(name.split()).lower()


def _normalize_role(role: str) -> str:
    """Normalize an agent role for matching against ROLE_TO_SPECIALIST.

    agents.yaml defines roles with YAML's `>` folded scalar syntax, which
    appends a trailing newline (e.g. "Billing Specialist\\n"). Collapsing
    whitespace and lowercasing strips that so it matches cleanly.

    Args:
        role: A role string, e.g. from a delegation tool's "coworker" arg.

    Returns:
        The role lowercased with whitespace collapsed to single spaces.
    """
    return " ".join(role.split()).lower()


@contextmanager
def track_tool_calls(tool_calls: List[str], delegations: Optional[List[str]] = None) -> Iterator[None]:
    """Temporarily listen for tool-usage events and record each tool name.

    Registers and unregisters only this one handler via crewai_event_bus.on()/
    .off(). Deliberately does NOT use crewai_event_bus.scoped_handlers(): that
    context manager clears every handler currently on the bus (not just ours)
    for the duration of the block, including CrewAI's own built-in listener
    (crewai.events.event_listener.EventListener) that prints the verbose
    Crew/Task/Agent console output - using it here silenced that output for
    the whole crew.kickoff() call.

    Args:
        tool_calls: List to append each invoked tool's name into, in order.
        delegations: Optional list to append each delegation target's role
            into, in order, whenever the manager calls "Delegate work to
            coworker" or "Ask question to coworker".

    Yields:
        None. Tool names (and delegation targets) are appended for the
        lifetime of the `with` block; the listener is removed (and only
        this one) on exit.
    """

    def _on_tool_usage(source: Any, event: ToolUsageStartedEvent) -> None:
        tool_calls.append(event.tool_name)
        if delegations is not None and _normalize_tool_name(event.tool_name) in DELEGATION_TOOL_NAMES:
            tool_args = event.tool_args if isinstance(event.tool_args, dict) else {}
            coworker = tool_args.get("coworker")
            if coworker:
                delegations.append(coworker)

    crewai_event_bus.on(ToolUsageStartedEvent)(_on_tool_usage)
    try:
        yield
    finally:
        crewai_event_bus.off(ToolUsageStartedEvent, _on_tool_usage)


def infer_agent(tool_calls: List[str], delegations: Optional[List[str]] = None) -> str:
    """Determine which specialist actually handled the query.

    Prefers the delegation target (who the manager explicitly handed the
    query to), since that's captured even when the specialist calls no
    tool of its own. Falls back to inferring from which specialist's tools
    fired when no delegation target was captured.

    Args:
        tool_calls: Tool names recorded by the tool-call tracker for one query.
        delegations: Coworker role names recorded whenever the manager
            delegated, in order.

    Returns:
        The most frequently targeted/invoked specialist ("billing", "tech",
        or "returns"), or "none" if neither signal identifies one.
    """
    if delegations:
        roles = [
            ROLE_TO_SPECIALIST[normalized]
            for role in delegations
            if (normalized := _normalize_role(role)) in ROLE_TO_SPECIALIST
        ]
        if roles:
            return Counter(roles).most_common(1)[0][0]

    specialists = [
        TOOL_TO_SPECIALIST[normalized]
        for name in tool_calls
        if (normalized := _normalize_tool_name(name)) in TOOL_TO_SPECIALIST
    ]
    if not specialists:
        return "none"
    return Counter(specialists).most_common(1)[0][0]


def log_handoff(
    query: str,
    tool_calls: List[str],
    response: str,
    delegations: Optional[List[str]] = None,
    agent_invoked: Optional[str] = None,
    log_path: Path = LOG_PATH,
) -> None:
    """Append one structured handoff record to the JSONL log.

    Args:
        query: The customer's raw support query.
        tool_calls: Tool names invoked while resolving the query, in order.
        response: The manager-reviewed, finalized response text.
        delegations: Coworker role names the manager delegated to, in order.
        agent_invoked: Which specialist handled the query. Defaults to
            infer_agent(tool_calls, delegations) when not given.
        log_path: Where to append the log line. Defaults to
            logs/handoff_log.jsonl at the project root.

    Returns:
        None. Creates the log file and its parent directory if missing, and
        appends one JSON object per line (JSONL), never overwriting prior runs.
    """
    delegations = delegations or []
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "agent_invoked": agent_invoked if agent_invoked is not None else infer_agent(tool_calls, delegations),
        "delegations": delegations,
        "tool_calls": tool_calls,
        "final_response": response,
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def read_last_handoff(log_path: Path = LOG_PATH) -> dict[str, Any]:
    """Read the most recently appended handoff record from the JSONL log.

    Lets callers (e.g. eval/run_eval.py) recover the tool_calls/agent_invoked
    that run_triage() already logged for a query, without tracking the same
    tool calls a second time.

    Args:
        log_path: Path to the JSONL log file.

    Returns:
        The last logged record, parsed as a dict.

    Raises:
        RuntimeError: If the log file doesn't exist yet or has no entries.
    """
    if not log_path.exists():
        raise RuntimeError(f"No handoff log found at {log_path}. Run a query through run_triage() first.")
    with open(log_path, encoding="utf-8") as f:
        lines = [line for line in f if line.strip()]
    if not lines:
        raise RuntimeError(f"Handoff log at {log_path} is empty.")
    return json.loads(lines[-1])
