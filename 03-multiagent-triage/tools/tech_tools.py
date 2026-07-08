"""Mock technical-support tools returning synthetic status/error data (no real backend)."""

from __future__ import annotations

from typing import Any

from crewai.tools import tool

_SYSTEM_STATUS: dict[str, dict[str, Any]] = {
    "api-gateway": {
        "status": "operational",
        "uptime_30d_pct": 99.98,
        "last_incident": "2026-05-12",
    },
    "auth-service": {
        "status": "degraded",
        "uptime_30d_pct": 97.42,
        "last_incident": "2026-07-06",
    },
    "billing-service": {
        "status": "operational",
        "uptime_30d_pct": 99.95,
        "last_incident": "2026-04-01",
    },
    "notification-service": {
        "status": "outage",
        "uptime_30d_pct": 91.20,
        "last_incident": "2026-07-07",
    },
}

_ERROR_CODES: dict[str, dict[str, str]] = {
    "ERR_AUTH_401": {
        "explanation": "Authentication failed - the request's credentials are missing or invalid.",
        "likely_cause": "Expired or malformed API token.",
        "suggested_action": "Ask the customer to re-authenticate or regenerate their API key.",
    },
    "ERR_RATE_LIMIT_429": {
        "explanation": "The client has sent too many requests in a given time window.",
        "likely_cause": "Request volume exceeded the account's rate limit.",
        "suggested_action": "Advise the customer to retry after backing off, or request a rate limit increase.",
    },
    "ERR_TIMEOUT_408": {
        "explanation": "The server timed out waiting for the request to complete.",
        "likely_cause": "Slow network connection or an overloaded downstream service.",
        "suggested_action": "Retry the request; escalate if timeouts persist across multiple attempts.",
    },
    "ERR_SERVER_500": {
        "explanation": "An unexpected internal server error occurred while processing the request.",
        "likely_cause": "Unhandled exception in the service backend.",
        "suggested_action": "Escalate to engineering with the request ID and timestamp.",
    },
}


@tool("Check System Status")
def check_system_status(service_name: str) -> dict[str, Any]:
    """Check the current operational status and 30-day uptime percentage for an internal
    service (e.g. 'api-gateway', 'auth-service', 'billing-service', 'notification-service').
    Use this when a customer reports something not working and you need to confirm whether
    it's a known outage."""
    status = _SYSTEM_STATUS.get(service_name)
    if status is None:
        return {
            "found": False,
            "service_name": service_name,
            "message": f"No status information found for service '{service_name}'.",
        }
    return {"found": True, "service_name": service_name, **status}


@tool("Lookup Error Code")
def lookup_error_code(error_code: str) -> dict[str, Any]:
    """Look up a canned explanation, likely cause, and suggested action for a known error
    code (e.g. 'ERR_AUTH_401', 'ERR_RATE_LIMIT_429'). Use this when a customer reports an
    error code from the product."""
    info = _ERROR_CODES.get(error_code)
    if info is None:
        return {
            "found": False,
            "error_code": error_code,
            "message": f"No information found for error code '{error_code}'.",
        }
    return {"found": True, "error_code": error_code, **info}
