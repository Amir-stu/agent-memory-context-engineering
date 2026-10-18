import json

from src.evaluator import EvaluationReport, ProbeResult, report_to_dict, report_to_markdown

HIT = ProbeResult("name", True, "Name is Zed")
MISS = ProbeResult("database", False, "")


def make_report(stale: tuple[str, ...] = ()) -> EvaluationReport:
    return EvaluationReport(
        memory_probes=(HIT, MISS),
        answer_probes=(HIT,),
        correction_probe=ProbeResult("Which database?", True, "Use DuckDB."),
        stale_conflicts=stale,
    )


def test_report_to_dict_contains_metrics_and_probe_details():
    data = report_to_dict(make_report(), model="model-x", run_at="2026-10-18T22:00:00")

    assert data["metrics"] == {
        "memory_recall": 0.5,
        "answer_recall": 1.0,
        "correction_applied": True,
        "stale_conflicts": 0,
        "passed": False,
    }
    assert data["model"] == "model-x"
    assert data["run_at"] == "2026-10-18T22:00:00"
    assert data["memory_probes"][1] == {"label": "database", "hit": False, "evidence": ""}
    json.dumps(data)


def test_report_to_markdown_renders_metric_table_with_percentages():
    table = report_to_markdown(make_report(), model="model-x", run_at="2026-10-18T22:00:00")

    assert "| Memory recall | 50% (1/2) |" in table
    assert "| Answer recall | 100% (1/1) |" in table
    assert "| Correction applied | yes |" in table
    assert "| Stale conflicts | 0 |" in table
    assert "model-x" in table
    assert "2026-10-18" in table


def test_report_to_markdown_when_stale_conflicts_lists_them():
    table = report_to_markdown(make_report(stale=("Uses PostgreSQL",)), model="m", run_at="2026-10-18")
    assert "| Stale conflicts | 1 |" in table
    assert "- Uses PostgreSQL" in table
