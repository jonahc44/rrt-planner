"""Smoke tests for the benchmark harness."""

import json

import pytest

from bench import run
from bench.scenarios import GENERATORS, generate, is_feasible


@pytest.mark.parametrize("family", list(GENERATORS))
def test_scenarios_are_deterministic_and_feasible(family):
    a, b = generate(family, 3, base_seed=7), generate(family, 3, base_seed=7)
    assert a == b
    assert is_feasible(a)
    assert generate(family, 4, base_seed=7) != a
    w = a.make_world()
    assert w.is_state_valid(list(a.start[:2])) and w.is_state_valid(list(a.goal[:2]))


def test_end_to_end_run_writes_results_and_plots(tmp_path):
    rc = run.main(["--families", "cluttered", "maze", "--instances", "1", "--runs", "2",
                   "--time-limit", "0.2", "--jobs", "4", "--out", str(tmp_path)])
    assert rc == 0
    data = json.loads((tmp_path / "results.json").read_text())
    assert len(data["runs"]) == 2 * 3 * 2
    r2 = [r for r in data["runs"] if r["space"] == "r2"]
    assert all(r["solved"] for r in r2)
    assert min(r["cost_ratio"] for r in r2) == pytest.approx(1.0)
    for r in data["runs"]:
        costs = [c for _, _, _, c in r["convergence"] if c is not None]
        assert costs == sorted(costs, reverse=True)
    for name in ["success_vs_time", "convergence", "tree_growth", "summary", "scenarios"]:
        assert (tmp_path / f"{name}.png").stat().st_size > 10_000
    assert "| cluttered |" in (tmp_path / "summary.md").read_text()
