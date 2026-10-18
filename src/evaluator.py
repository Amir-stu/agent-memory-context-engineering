"""Cross-session memory evaluation.

Scenario: state facts in session 1, delete that thread, ask for help in session 2 without
restating anything, correct one preference, then ask again from a brand-new thread so the
answer can only come from long-term memory. Scoring is keyword-based on purpose: it is
deterministic and cheap, at the cost of missing paraphrases (see README, "Limitations").
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from src.agent_graph import CompanionAgent
from src.memory_manager import MemoryRecord


@dataclass(frozen=True)
class Probe:
    label: str
    keywords: tuple[str, ...]


@dataclass(frozen=True)
class Scenario:
    user_id: str
    session_one_thread: str
    session_two_thread: str
    follow_up_thread: str
    session_one: tuple[str, ...]
    stored_facts: tuple[Probe, ...]
    session_two: tuple[Probe, ...]
    correction: str
    after_correction: Probe
    stale_keywords: tuple[str, ...]
    fresh_keywords: tuple[str, ...]


@dataclass(frozen=True)
class ProbeResult:
    label: str
    hit: bool
    evidence: str


@dataclass(frozen=True)
class EvaluationReport:
    memory_probes: tuple[ProbeResult, ...]
    answer_probes: tuple[ProbeResult, ...]
    correction_probe: ProbeResult
    stale_conflicts: tuple[str, ...]

    @property
    def memory_recall(self) -> float:
        return _hit_rate(self.memory_probes)

    @property
    def answer_recall(self) -> float:
        return _hit_rate(self.answer_probes)

    @property
    def correction_applied(self) -> bool:
        return self.correction_probe.hit

    @property
    def passed(self) -> bool:
        return (
            self.memory_recall == 1.0
            and self.answer_recall == 1.0
            and self.correction_applied
            and not self.stale_conflicts
        )


# Obviously fictional persona; nothing here describes a real person.
DEFAULT_SCENARIO = Scenario(
    user_id="eval-zed",
    session_one_thread="session-1",
    session_two_thread="session-2",
    follow_up_thread="session-3",
    session_one=(
        "Hi, I'm Zed Testperson. I work as a data analyst at a fictional company called Example Corp.",
        "My career goal is to become an ML platform engineer within two years.",
        "For technical work I prefer Python, and PostgreSQL is my go-to database.",
    ),
    stored_facts=(
        Probe("name", ("zed",)),
        Probe("current role", ("analyst",)),
        Probe("career goal", ("ml platform",)),
        Probe("language", ("python",)),
        Probe("database", ("postgres",)),
    ),
    session_two=(
        Probe("What's my name, and what do I do for work?", ("zed",)),
        Probe(
            "Suggest one skill I should learn next for my career goal.", ("ml platform", "mlops", "machine learning")
        ),
        Probe("Write a two-line Python snippet that connects to the database I usually use.", ("postgres", "psycopg")),
    ),
    correction="Update: I've moved off PostgreSQL completely. DuckDB is now my only database.",
    after_correction=Probe("Which database should my next side project use?", ("duckdb",)),
    stale_keywords=("postgres",),
    fresh_keywords=("duckdb",),
)


def mentions(text: str, keywords: Sequence[str]) -> bool:
    lowered = text.casefold()
    return any(keyword.casefold() in lowered for keyword in keywords)


def _hit_rate(results: Sequence[ProbeResult]) -> float:
    return sum(result.hit for result in results) / len(results) if results else 0.0


def _settle(agent: CompanionAgent) -> None:
    errors = agent.memory.wait_for_pending()
    if errors:
        raise RuntimeError(f"memory write failed during evaluation: {errors[0]}") from errors[0]


def _probe_memories(memories: Sequence[MemoryRecord], probe: Probe) -> ProbeResult:
    match = next((memory.text for memory in memories if mentions(memory.text, probe.keywords)), "")
    return ProbeResult(probe.label, bool(match), match)


def _ask(agent: CompanionAgent, scenario: Scenario, thread: str, probe: Probe, log: Callable) -> ProbeResult:
    answer = agent.reply(scenario.user_id, thread, probe.label).answer
    log(thread, probe.label, answer)
    return ProbeResult(probe.label, mentions(answer, probe.keywords), answer)


def _no_log(thread: str, question: str, answer: str) -> None:
    return None


def run_scenario(
    agent: CompanionAgent,
    scenario: Scenario,
    log: Callable[[str, str, str], None] = _no_log,
) -> EvaluationReport:
    for message in scenario.session_one:
        log(
            scenario.session_one_thread,
            message,
            agent.reply(scenario.user_id, scenario.session_one_thread, message).answer,
        )
    _settle(agent)
    stored = agent.memory.list_memories(scenario.user_id)
    memory_probes = tuple(_probe_memories(stored, probe) for probe in scenario.stored_facts)

    agent.clear_thread(scenario.user_id, scenario.session_one_thread)
    answer_probes = tuple(
        _ask(agent, scenario, scenario.session_two_thread, probe, log) for probe in scenario.session_two
    )

    correction_answer = agent.reply(scenario.user_id, scenario.session_two_thread, scenario.correction).answer
    log(scenario.session_two_thread, scenario.correction, correction_answer)
    _settle(agent)
    correction_probe = _ask(agent, scenario, scenario.follow_up_thread, scenario.after_correction, log)
    _settle(agent)

    final_memories = agent.memory.list_memories(scenario.user_id)
    stale_conflicts = tuple(
        memory.text
        for memory in final_memories
        if mentions(memory.text, scenario.stale_keywords) and not mentions(memory.text, scenario.fresh_keywords)
    )
    return EvaluationReport(memory_probes, answer_probes, correction_probe, stale_conflicts)
