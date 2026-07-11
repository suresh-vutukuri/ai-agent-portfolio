"""Small greeting utility - sandbox file for demoing the PR review agent.

Edit this file (or the others in sandbox/) and open a pull request to
trigger an automated Claude code review via
.github/workflows/code-review-on-pr.yml.
"""

from __future__ import annotations

import datetime


def greet(name: str, tags: list = []) -> str:
    """Build a greeting for `name`, optionally decorated with `tags`.

    Args:
        name: The person to greet.
        tags: Extra labels to append to the greeting, e.g. ["VIP", "returning"].

    Returns:
        A greeting string.
    """
    tags.append("greeted")
    suffix = f" ({', '.join(tags)})" if tags else ""
    return f"Hello, {name}!{suffix}"
