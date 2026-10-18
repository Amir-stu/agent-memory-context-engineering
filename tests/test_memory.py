"""Cross-session retention: state facts, end the session, come back and get help without restating.

Offline tests run the full scenario against deterministic boundaries (fake LLM + fake mem0)
and prove the plumbing: memory survives a cleared thread and reaches the next prompt.
The `live` test runs the same scenario against real mem0 + a real LLM, which is the only
place fact *correction* can be judged, because reconciliation is mem0's LLM step.
"""

import os

import pytest
from dotenv import load_dotenv

from src.evaluator import DEFAULT_SCENARIO, mentions, run_scenario


def test_run_scenario_when_offline_recalls_session_one_facts_in_session_two(agent):
    report = run_scenario(agent, DEFAULT_SCENARIO)

    assert report.memory_recall == 1.0
    assert report.answer_recall == 1.0
    assert all(probe.hit for probe in report.answer_probes)


def test_run_scenario_when_offline_session_two_runs_on_cleared_thread(agent):
    run_scenario(agent, DEFAULT_SCENARIO)
    assert agent.history(DEFAULT_SCENARIO.user_id, DEFAULT_SCENARIO.session_one_thread) == []


def test_run_scenario_when_backend_does_not_reconcile_reports_stale_conflict(agent):
    report = run_scenario(agent, DEFAULT_SCENARIO)

    assert report.stale_conflicts
    assert report.correction_applied is True
    assert report.passed is False


def test_mentions_when_keyword_differs_in_case_matches():
    assert mentions("I use DuckDB daily", ("duckdb",))


def test_mentions_when_no_keyword_present_returns_false():
    assert not mentions("I use SQLite", ("duckdb", "postgres"))


@pytest.mark.live
def test_run_scenario_when_live_retains_facts_and_applies_correction(tmp_path):
    load_dotenv()
    if not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY")):
        pytest.skip("live evaluation needs an LLM API key")
    from src.agent_graph import create_agent
    from src.config import load_settings

    settings = load_settings({**os.environ, "DATA_DIR": str(tmp_path)})
    agent = create_agent(settings)
    try:
        report = run_scenario(agent, DEFAULT_SCENARIO)
    finally:
        agent.close()

    assert report.memory_recall >= 0.8
    assert report.answer_recall >= 0.75
    assert report.correction_applied
    assert not report.stale_conflicts


def test_run_scenario_when_memory_write_fails_raises_instead_of_scoring(chat_model, settings):
    from langgraph.checkpoint.memory import InMemorySaver

    from src.agent_graph import CompanionAgent
    from src.memory_manager import MemoryManager
    from tests.fakes import ExplodingMem0

    agent = CompanionAgent(chat_model, MemoryManager(ExplodingMem0()), InMemorySaver(), settings)
    with pytest.raises(RuntimeError, match="memory write failed"):
        run_scenario(agent, DEFAULT_SCENARIO)
