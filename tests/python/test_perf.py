"""Tests for the performance-regression tooling."""

import copy
import json

import pytest

from bench import perf_ab, perf_compare


def case(cid, planner, us, work=10.0, cost=5.0, solved=True):
    return {"id": cid, "planner": planner, "space": "r2", "iterations": 100,
            "time_per_iter": us * 1e-6, "time_samples": [us * 1e-6],
            "solved": solved, "cost": cost if solved else None, "tree_size": 50,
            "work": {"distance_calls": work, "connect_calls": work, "steer_calls": 1,
                     "state_checks": 0, "motion_checks": work, "checked_length": work}}


def write(tmp_path, name, cases, counters=True):
    p = tmp_path / f"{name}.json"
    p.write_text(json.dumps({"meta": {"label": name, "has_counters": counters,
                                      "iterations": {"r2": 100}},
                             "cases": cases}))
    return p


BASE = [case(f"c{i}", "A" if i < 4 else "B", 10.0 + i) for i in range(8)]


def run_compare(tmp_path, head_cases, *extra):
    b = write(tmp_path, "base", BASE)
    h = write(tmp_path, "head", head_cases)
    return perf_compare.main(["--base", str(b), "--head", str(h),
                              "--report", str(tmp_path / "r.md"), *extra])


def test_identical_builds_pass(tmp_path):
    assert run_compare(tmp_path, copy.deepcopy(BASE)) == 0
    assert "No regression" in (tmp_path / "r.md").read_text()


def test_small_noise_passes(tmp_path):
    head = copy.deepcopy(BASE)
    for c in head:
        c["time_per_iter"] *= 1.04
    assert run_compare(tmp_path, head) == 0


def test_time_regression_fails(tmp_path):
    head = copy.deepcopy(BASE)
    for c in head:
        c["time_per_iter"] *= 1.2
    assert run_compare(tmp_path, head) == 1
    assert "time/iteration" in (tmp_path / "r.md").read_text()


def test_single_planner_time_regression_fails(tmp_path):
    head = copy.deepcopy(BASE)
    for c in head:
        if c["planner"] == "B":
            c["time_per_iter"] *= 1.25
    assert run_compare(tmp_path, head) == 1
    assert "B: time/iteration" in (tmp_path / "r.md").read_text()


def test_cheap_noisy_case_cannot_fail_the_gate(tmp_path):
    # Time is a ratio of totals: a 0.5 us case doubling is noise, not signal.
    base = BASE + [case("tiny", "A", 0.5)]
    head = copy.deepcopy(base)
    head[-1]["time_per_iter"] *= 2
    b, h = write(tmp_path, "base", base), write(tmp_path, "head", head)
    assert perf_compare.main(["--base", str(b), "--head", str(h)]) == 0


def test_work_regression_fails_even_when_time_is_flat(tmp_path):
    head = copy.deepcopy(BASE)
    for c in head:
        c["work"] = {k: v * 1.3 for k, v in c["work"].items()}
    assert run_compare(tmp_path, head) == 1
    assert "work/iteration" in (tmp_path / "r.md").read_text()


def test_cost_and_success_regressions_fail(tmp_path):
    head = copy.deepcopy(BASE)
    for c in head:
        c["cost"] *= 1.1
    assert run_compare(tmp_path, head) == 1
    head = copy.deepcopy(BASE)
    for c in head[:3]:
        c["solved"], c["cost"] = False, None
    assert run_compare(tmp_path, head) == 1
    assert "success" in (tmp_path / "r.md").read_text()


def test_allow_regression_reports_but_passes(tmp_path):
    head = copy.deepcopy(BASE)
    for c in head:
        c["time_per_iter"] *= 2
    assert run_compare(tmp_path, head, "--allow-regression") == 0
    assert "Regression detected" in (tmp_path / "r.md").read_text()


def test_missing_counters_skips_work(tmp_path):
    head = copy.deepcopy(BASE)
    for c in head:
        c.pop("work")
    b = write(tmp_path, "base", BASE)
    h = write(tmp_path, "head", head, counters=False)
    assert perf_compare.main(["--base", str(b), "--head", str(h),
                              "--report", str(tmp_path / "r.md")]) == 0
    assert "Work counters unavailable" in (tmp_path / "r.md").read_text()


def test_multiple_files_keep_the_fastest_time(tmp_path):
    slow = copy.deepcopy(BASE)
    for c in slow:
        c["time_per_iter"] *= 3
    b1 = write(tmp_path, "b1", slow)
    b2 = write(tmp_path, "b2", BASE)
    _, merged = perf_compare.load([b1, b2])
    assert [c["time_per_iter"] for c in merged] == [c["time_per_iter"] for c in BASE]


@pytest.mark.slow
def test_ab_driver_head_vs_itself(tmp_path):
    rc = perf_ab.main(["--base-path", "python", "--head-path", "python",
                       "--rounds", "1", "--quick", "--out-dir", str(tmp_path)])
    assert rc == 0
    base = json.loads((tmp_path / "base.json").read_text())
    head = json.loads((tmp_path / "head.json").read_text())
    assert len(base["cases"]) == len(head["cases"]) > 0
    # Same code, same seeds: identical work and quality.
    for b, h in zip(base["cases"], head["cases"]):
        assert b["id"] == h["id"]
        assert b["work"] == h["work"]
        assert b["cost"] == h["cost"]


def test_ab_driver_reports_missing_baseline(tmp_path):
    rc = perf_ab.main(["--base-path", str(tmp_path / "nope"), "--head-path", "python",
                       "--rounds", "1", "--quick", "--out-dir", str(tmp_path)])
    assert rc == 3
    assert (tmp_path / "head.json").exists() and not (tmp_path / "base.json").exists()
