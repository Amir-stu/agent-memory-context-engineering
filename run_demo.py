"""Terminal demo: cross-session recall and fact correction against the real LLM + mem0.

python run_demo.py            # fresh user id each run, so earlier runs don't leak in
python run_demo.py --user-id eval-zed
"""

import argparse
import sys
import textwrap
from dataclasses import replace
from datetime import datetime

from dotenv import load_dotenv

from src.agent_graph import create_agent
from src.config import load_settings
from src.evaluator import DEFAULT_SCENARIO, EvaluationReport, ProbeResult, run_scenario

WRAP_WIDTH = 100
ANSWER_PREVIEW_CHARS = 400


def print_turn(thread: str, message: str, answer: str) -> None:
    preview = answer if len(answer) <= ANSWER_PREVIEW_CHARS else answer[:ANSWER_PREVIEW_CHARS] + " ..."
    print(f"\n[{thread}] USER: {message}")
    print(textwrap.fill(preview, WRAP_WIDTH, initial_indent="  AGENT: ", subsequent_indent=" " * 9))


def print_probes(title: str, probes: tuple[ProbeResult, ...]) -> None:
    print(f"\n{title}")
    for probe in probes:
        status = "PASS" if probe.hit else "MISS"
        print(f"  [{status}] {probe.label}")


def print_report(report: EvaluationReport) -> None:
    print("\n" + "=" * WRAP_WIDTH)
    print_probes("Facts extracted into long-term memory after session 1:", report.memory_probes)
    print_probes("Session 2 answers (thread 1 deleted, nothing restated):", report.answer_probes)
    print_probes("Answer after correction, asked from a brand-new thread:", (report.correction_probe,))
    print(f"\nmemory recall      {report.memory_recall:.0%}")
    print(f"answer recall      {report.answer_recall:.0%}")
    print(f"correction applied {report.correction_applied}")
    print(f"stale conflicts    {list(report.stale_conflicts) or 'none'}")
    print(f"\nRESULT: {'PASS' if report.passed else 'FAIL'}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--user-id", help="memory namespace to use (default: a fresh one per run)")
    args = parser.parse_args()

    load_dotenv()
    user_id = args.user_id or f"demo-{datetime.now():%Y%m%d-%H%M%S}"
    scenario = replace(DEFAULT_SCENARIO, user_id=user_id)
    agent = create_agent(load_settings())
    print(f"user_id = {user_id}")
    try:
        report = run_scenario(agent, scenario, log=print_turn)
        print_report(report)
        print("\nWhat the agent remembers now:")
        for memory in agent.memory.list_memories(user_id):
            print(f"  - {memory.text}")
    finally:
        agent.close()
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
