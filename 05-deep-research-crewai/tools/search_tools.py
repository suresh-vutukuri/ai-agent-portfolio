"""CrewAI tool wrapping Tavily web search."""

from __future__ import annotations

import os
from typing import Any

from crewai.tools import tool
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

_MAX_RESULTS = 5

# Records every web_search call made since the last reset_search_transcript(),
# as {"query": str, "results": list[dict]}. Lets the eval harness recover the
# original Tavily results a report's citations were (supposedly) drawn from,
# without re-running searches or parsing CrewAI's internal event bus.
_search_transcript: list[dict[str, Any]] = []


def _get_client() -> TavilyClient:
    """Build a TavilyClient using TAVILY_API_KEY from the environment/.env.

    Returns:
        A configured TavilyClient.

    Raises:
        RuntimeError: If TAVILY_API_KEY is not set.
    """
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY is not set. Add it to your .env file.")
    return TavilyClient(api_key=api_key)


@tool("Web Search")
def web_search(query: str) -> list[dict[str, Any]]:
    """Search the web via Tavily and return the top results for a query. Each result
    includes its title, URL, and a short snippet. Use this to find sourced facts for a
    sub-question rather than relying on prior knowledge alone."""
    client = _get_client()
    response = client.search(query=query, max_results=_MAX_RESULTS)
    results = [
        {
            "title": result.get("title", ""),
            "url": result.get("url", ""),
            "snippet": result.get("content", ""),
        }
        for result in response.get("results", [])
    ]
    _search_transcript.append({"query": query, "results": results})
    return results


def reset_search_transcript() -> None:
    """Clear the recorded log of web_search calls. Call before kicking off a crew
    run whose search transcript you intend to inspect afterward."""
    _search_transcript.clear()


def get_search_transcript() -> list[dict[str, Any]]:
    """Return every web_search call recorded since the last reset.

    Returns:
        A list of {"query": str, "results": list[dict]} entries, in call order.
    """
    return [dict(entry) for entry in _search_transcript]


def build_url_content_index(transcript: list[dict[str, Any]]) -> dict[str, str]:
    """Flatten a search transcript into a URL -> original snippet index.

    Args:
        transcript: A search transcript as returned by get_search_transcript().

    Returns:
        A dict mapping each result URL to the snippet Tavily returned for it.
        If the same URL appears in multiple searches, the first snippet seen
        is kept.
    """
    index: dict[str, str] = {}
    for entry in transcript:
        for result in entry.get("results", []):
            url = result.get("url")
            if url and url not in index:
                index[url] = result.get("snippet", "")
    return index
