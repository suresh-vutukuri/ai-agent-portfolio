"""Sequential CrewAI crew: plans a topic into sub-questions, researches each via
web search, and synthesizes a cited markdown report."""

from __future__ import annotations

import sys
from pathlib import Path

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# tools/ is a plain sibling dir, not a package
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from search_tools import web_search  # noqa: E402


@CrewBase
class ResearchCrew:
    """Plans, researches, and synthesizes a cited markdown report for a topic."""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def planner_agent(self) -> Agent:
        """Research Planner: breaks the topic into 3-5 distinct sub-questions."""
        return Agent(config=self.agents_config["planner_agent"])

    @agent
    def researcher_agent(self) -> Agent:
        """Web Researcher, equipped with the web_search tool."""
        return Agent(config=self.agents_config["researcher_agent"], tools=[web_search])

    @agent
    def writer_agent(self) -> Agent:
        """Synthesis Writer: compiles findings into a cited markdown report."""
        return Agent(config=self.agents_config["writer_agent"])

    @task
    def plan_task(self) -> Task:
        """Breaks the topic into 3-5 sub-questions."""
        return Task(config=self.tasks_config["plan_task"], agent=self.planner_agent())

    @task
    def research_task(self) -> Task:
        """Researches each sub-question with the web_search tool, sourced findings."""
        return Task(
            config=self.tasks_config["research_task"],
            agent=self.researcher_agent(),
            context=[self.plan_task()],
        )

    @task
    def synthesize_task(self) -> Task:
        """Compiles sourced findings into a cited markdown report."""
        return Task(
            config=self.tasks_config["synthesize_task"],
            agent=self.writer_agent(),
            context=[self.research_task()],
        )

    @crew
    def crew(self) -> Crew:
        """Assembles the sequential research crew: plan -> research -> synthesize."""
        return Crew(
            agents=[self.planner_agent(), self.researcher_agent(), self.writer_agent()],
            tasks=[self.plan_task(), self.research_task(), self.synthesize_task()],
            process=Process.sequential,
            verbose=True,
        )
