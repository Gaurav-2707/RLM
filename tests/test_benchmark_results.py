import json

import pytest

from RLM.experiments.benchmark_suite import run
from RLM.experiments.summarize_results import summarize


def test_benchmark_summary_is_derived_from_traces(tmp_path):
    out_dir = tmp_path / "yc_proof"
    run(task_count=100, out_dir=str(out_dir))

    rows = summarize(str(out_dir))

    full = next(row for row in rows if row["Method"] == "RLM-1 (Full Neuro-Symbolic System)")
    no_memory = next(row for row in rows if row["Method"] == "RLM-1 (No Episodic Memory)")
    assert full["Accuracy"] == "86%"
    assert no_memory["Accuracy"] == "54%"
    assert int(full["Failure Loops"]) < int(no_memory["Failure Loops"])


def test_summary_rejects_incomplete_traces(tmp_path):
    bad_dir = tmp_path / "bad"
    bad_dir.mkdir()
    (bad_dir / "trace.json").write_text(json.dumps({"metadata": {"method": "x"}}))

    with pytest.raises(ValueError):
        summarize(str(bad_dir))
