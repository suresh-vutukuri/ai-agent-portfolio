"""Mock billing tools returning synthetic invoice/payment data (no real backend)."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from crewai.tools import tool

_TODAY = date.today()

_INVOICES: dict[str, dict[str, Any]] = {
    "INV-5001": {
        "amount": 129.99,
        "due_date": (_TODAY + timedelta(days=10)).isoformat(),
        "status": "unpaid",
    },
    "INV-5002": {
        "amount": 59.00,
        "due_date": (_TODAY - timedelta(days=5)).isoformat(),
        "status": "overdue",
    },
    "INV-5003": {
        "amount": 249.50,
        "due_date": (_TODAY - timedelta(days=20)).isoformat(),
        "status": "paid",
    },
    "INV-5004": {
        "amount": 15.00,
        "due_date": (_TODAY + timedelta(days=30)).isoformat(),
        "status": "unpaid",
    },
}

_PAYMENT_HISTORY: dict[str, list[dict[str, Any]]] = {
    "CUST-2001": [
        {"date": (_TODAY - timedelta(days=95)).isoformat(), "amount": 59.00, "method": "credit_card", "status": "completed"},
        {"date": (_TODAY - timedelta(days=65)).isoformat(), "amount": 59.00, "method": "credit_card", "status": "completed"},
        {"date": (_TODAY - timedelta(days=35)).isoformat(), "amount": 59.00, "method": "credit_card", "status": "completed"},
    ],
    "CUST-2002": [
        {"date": (_TODAY - timedelta(days=200)).isoformat(), "amount": 249.50, "method": "paypal", "status": "completed"},
        {"date": (_TODAY - timedelta(days=110)).isoformat(), "amount": 249.50, "method": "paypal", "status": "completed"},
        {"date": (_TODAY - timedelta(days=20)).isoformat(), "amount": 249.50, "method": "paypal", "status": "failed"},
    ],
    "CUST-2003": [
        {"date": (_TODAY - timedelta(days=400)).isoformat(), "amount": 15.00, "method": "bank_transfer", "status": "completed"},
    ],
}


@tool("Check Invoice Status")
def check_invoice_status(invoice_id: str) -> dict[str, Any]:
    """Look up a billing invoice by its ID and return its amount, due date, and payment
    status (paid, unpaid, or overdue). Use this when a customer asks about a specific
    invoice or bill."""
    invoice = _INVOICES.get(invoice_id)
    if invoice is None:
        return {
            "found": False,
            "invoice_id": invoice_id,
            "message": f"No invoice found with id '{invoice_id}'.",
        }
    return {"found": True, "invoice_id": invoice_id, **invoice}


@tool("Get Payment History")
def get_payment_history(customer_id: str) -> dict[str, Any]:
    """Retrieve a customer's most recent payments (up to the last 3), including date,
    amount, method, and status. Use this when a customer asks about their payment or
    billing history."""
    payments = _PAYMENT_HISTORY.get(customer_id)
    if payments is None:
        return {
            "found": False,
            "customer_id": customer_id,
            "message": f"No payment history found for customer '{customer_id}'.",
        }
    return {"found": True, "customer_id": customer_id, "payments": payments[-3:]}
