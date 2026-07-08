"""Hierarchical CrewAI crew: a manager triages a query and delegates to a specialist."""

from __future__ import annotations

import sys
from pathlib import Path

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# tools/ is a plain sibling dir, not a package
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from billing_tools import check_invoice_status, get_payment_history  # noqa: E402
from returns_tools import check_refund_eligibility, lookup_order  # noqa: E402
from tech_tools import check_system_status, lookup_error_code  # noqa: E402


@CrewBase
class TriageCrew:
    """Routes a customer query to the right specialist and finalizes their response."""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def manager_agent(self) -> Agent:
        """Triage Manager: classifies the query, delegates it, and reviews the result.

        Never added to the crew's `agents` list - passed as `manager_agent` to
        Crew() instead, per CrewAI's hierarchical process convention.
        """
        return Agent(config=self.agents_config["manager_agent"])

    @agent
    def billing_agent(self) -> Agent:
        """Billing Specialist, equipped with invoice and payment-history tools."""
        return Agent(
            config=self.agents_config["billing_agent"],
            tools=[check_invoice_status, get_payment_history],
        )

    @agent
    def tech_agent(self) -> Agent:
        """Technical Support Specialist, equipped with status and error-code tools."""
        return Agent(
            config=self.agents_config["tech_agent"],
            tools=[check_system_status, lookup_error_code],
        )

    @agent
    def returns_agent(self) -> Agent:
        """Returns Specialist, equipped with order-lookup and refund-eligibility tools."""
        return Agent(
            config=self.agents_config["returns_agent"],
            tools=[lookup_order, check_refund_eligibility],
        )

    @task
    def triage_task(self) -> Task:
        """Classifies the incoming query as billing, technical, or returns."""
        return Task(config=self.tasks_config["triage_task"])

    @task
    def resolve_task(self) -> Task:
        """Specialist resolves the query with tools; the manager reviews and finalizes."""
        return Task(
            config=self.tasks_config["resolve_task"],
            context=[self.triage_task()],
        )

    @crew
    def crew(self) -> Crew:
        """Assembles the hierarchical crew: the manager delegates to the specialists.

        The manager is passed via `manager_agent`, not included in `agents`, and
        neither task is pre-assigned an agent - the manager decides who handles
        each one at run time.
        """
        return Crew(
            agents=[self.billing_agent(), self.tech_agent(), self.returns_agent()],
            tasks=[self.triage_task(), self.resolve_task()],
            process=Process.hierarchical,
            manager_agent=self.manager_agent(),
            verbose=True,
        )
