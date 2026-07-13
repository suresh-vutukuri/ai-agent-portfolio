"""Shared graph state definition for the trading research copilot pipeline."""

from __future__ import annotations

from typing import Optional, TypedDict

import pandas as pd


class CopilotState(TypedDict):
    """State threaded through the LangGraph pipeline.

    Fields are populated progressively as the graph runs: `ticker` is set at
    the start; `htf_df`, `ltf_df`, `htf_bias`, and `ltf_candidates` are filled
    in by the fetch/analysis nodes in sequence; `synthesis_output` is filled
    in last by the synthesis LLM node. `errors` accumulates human-readable
    messages from any node that failed, instead of raising and aborting the
    whole run (see nodes.py).
    """

    ticker: str
    htf_df: Optional[pd.DataFrame]
    ltf_df: Optional[pd.DataFrame]
    htf_bias: Optional[dict]
    ltf_candidates: Optional[list[dict]]
    synthesis_output: Optional[str]
    errors: list[str]
