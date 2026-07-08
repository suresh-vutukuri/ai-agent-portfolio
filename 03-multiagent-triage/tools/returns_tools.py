"""Mock returns/orders tools returning synthetic order data (no real backend)."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from crewai.tools import tool

REFUND_WINDOW_DAYS = 30

_TODAY = date.today()

_ORDERS: dict[str, dict[str, Any]] = {
    "ORD-1001": {
        "items": ["Wireless Mouse", "USB-C Cable"],
        "order_date": (_TODAY - timedelta(days=5)).isoformat(),
        "status": "delivered",
        "total": 42.98,
    },
    "ORD-1002": {
        "items": ["Mechanical Keyboard"],
        "order_date": (_TODAY - timedelta(days=45)).isoformat(),
        "status": "delivered",
        "total": 89.99,
    },
    "ORD-1003": {
        "items": ["27-inch Monitor", "HDMI Cable"],
        "order_date": (_TODAY - timedelta(days=2)).isoformat(),
        "status": "shipped",
        "total": 249.50,
    },
    "ORD-1004": {
        "items": ["Laptop Stand"],
        "order_date": (_TODAY - timedelta(days=90)).isoformat(),
        "status": "cancelled",
        "total": 34.00,
    },
}


@tool("Lookup Order")
def lookup_order(order_id: str) -> dict[str, Any]:
    """Look up an order by its ID and return its items, order date, status (delivered,
    shipped, processing, or cancelled), and total. Use this when a customer asks about
    the contents or status of a specific order."""
    order = _ORDERS.get(order_id)
    if order is None:
        return {
            "found": False,
            "order_id": order_id,
            "message": f"No order found with id '{order_id}'.",
        }
    return {"found": True, "order_id": order_id, **order}


@tool("Check Refund Eligibility")
def check_refund_eligibility(order_id: str) -> dict[str, Any]:
    """Determine whether an order is eligible for a refund, based on how many days have
    passed since purchase and the order's current status. Use this when a customer asks
    if they can get a refund or return an item."""
    order = _ORDERS.get(order_id)
    if order is None:
        return {
            "found": False,
            "order_id": order_id,
            "message": f"No order found with id '{order_id}'.",
        }

    order_date = date.fromisoformat(order["order_date"])
    days_since_purchase = (_TODAY - order_date).days

    if order["status"] == "cancelled":
        eligible = False
        reason = "Order was cancelled; there is nothing to refund."
    elif order["status"] != "delivered":
        eligible = False
        reason = f"Order status is '{order['status']}'; refunds are only available after delivery."
    elif days_since_purchase <= REFUND_WINDOW_DAYS:
        eligible = True
        reason = f"Purchased {days_since_purchase} day(s) ago, within the {REFUND_WINDOW_DAYS}-day refund window."
    else:
        eligible = False
        reason = f"Purchased {days_since_purchase} day(s) ago, past the {REFUND_WINDOW_DAYS}-day refund window."

    return {
        "found": True,
        "order_id": order_id,
        "days_since_purchase": days_since_purchase,
        "eligible": eligible,
        "reason": reason,
    }
