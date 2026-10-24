"""Performance-regression suite: fixed work, deterministic seeds.

    python -m bench.perf --out perf.json [--repeats 2]

Unlike bench.run (fixed *time*, measures quality), every case here runs a
fixed number of iterations with a fixed seed, so two builds of the same
code do identical work. Per case it records:

  time_per_iter   best of --repeats timed solves (no instrumentation)
  work            calls per iteration into the StateSpace / CollisionChecker,
                  counted with CountingStateSpace / CountingCollisionChecker.
                  Deterministic, so it catches algorithmic regressions with
                  zero noise even on a shared CI runner.
  quality         solved? and final cost at the iteration budget.

Compare two result files with bench.perf_compare. Runs against whatever
rrt_planner is importable, so CI can point PYTHONPATH at a base-branch
build and reuse this script and its scenarios. For the most robust A/B
timing use bench.perf_ab, which interleaves the two builds case by case.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import rrt_planner as rp

from .planners import PLANNERS
from .scenarios import GENERATORS, generate

#: Iteration budget per planner space: about 0.2 s per solve on a laptop.
ITERATIONS = {"r2": 8000, "dubins": 1500}
FAMILIES = list(GENERATORS)
INSTANCES = 2
SEEDS = (1, 2)
SCENARIO_SEED = 1234  # independent of bench.run's default suite

HAS_COUNTERS = hasattr(rp, "CountingStateSpace")


def _case_id(scn, spec, seed) -> str:
    return f"{scn.name}/{spec.name}/s{seed}"


def _build(spec, scn, seed, counting: bool):
    planner, problem = spec.build(scn, seed)
    if not counting:
        return planner, None, None
    space = rp.CountingStateSpace(problem.space)
    checker = rp.CountingCollisionChecker(problem.checker)
    goal = problem.goal
    counted = rp.PlanningProblem(space=space, checker=checker, cost=problem.cost,
                                 start=problem.start,
                                 goal=rp.GoalState(space, goal.state, goal.tolerance))
    planner.set_problem(counted)
    return planner, space, checker


def run_case(scn, spec, seed: int, repeats: int, work: bool = True) -> dict:
    iters = ITERATIONS[spec.space]
    case = {"id": _case_id(scn, spec, seed), "family": scn.family,
            "instance": scn.instance, "planner": spec.name, "space": spec.space,
            "seed": seed, "iterations": iters}

    # Work + quality pass (instrumented; identical trajectory to the timed runs).
    if work:
        planner, space, checker = _build(spec, scn, seed, counting=HAS_COUNTERS)
        res = planner.solve(max_iterations=iters)
        case.update(solved=res.solved, cost=res.cost if res.solved else None,
                    tree_size=res.tree_size)
    if work and HAS_COUNTERS:
        case["work"] = {
            "distance_calls": space.distance_calls / iters,
            "connect_calls": space.connect_calls / iters,
            "steer_calls": space.steer_calls / iters,
            "state_checks": checker.state_calls / iters,
            "motion_checks": checker.motion_calls / iters,
            "checked_length": checker.checked_length / iters,
        }

    # Timed passes: fresh planner each time, best-of to reject interference.
    samples = []
    for _ in range(repeats):
        planner, _, _ = _build(spec, scn, seed, counting=False)
        samples.append(planner.solve(max_iterations=iters).elapsed / iters)
    case["time_per_iter"] = min(samples)
    case["time_samples"] = samples
    return case


def suite(quick: bool = False):
    """The fixed list of (scenario, planner spec, seed) cases."""
    instances = 1 if quick else INSTANCES
    seeds = SEEDS[:1] if quick else SEEDS
    scenarios = [generate(f, i, SCENARIO_SEED) for f in FAMILIES for i in range(instances)]
    return [(s, spec, seed) for s in scenarios for spec in PLANNERS.values()
            for seed in seeds]


def meta(label: str | None, repeats: int) -> dict:
    return {
        "label": label or _git_sha() or "unknown",
        "git_sha": _git_sha(),
        "module": rp.__file__,
        "has_counters": HAS_COUNTERS,
        "iterations": ITERATIONS,
        "repeats": repeats,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
    }


def worker(label: str | None, quick: bool = False) -> int:
    """Line-oriented JSON protocol for bench.perf_ab: announce the suite,
    then answer {"id", "repeats", "work"} requests until stdin closes."""
    cases = {_case_id(*c): c for c in suite(quick)}
    print(json.dumps({"meta": meta(label, 0), "cases": list(cases)}), flush=True)
    for line in sys.stdin:
        req = json.loads(line)
        scn, spec, seed = cases[req["id"]]
        res = run_case(scn, spec, seed, req["repeats"], work=req.get("work", True))
        print(json.dumps(res), flush=True)
    return 0


def _git_sha() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True,
                              cwd=Path(rp.__file__).parent).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--worker", action="store_true",
                    help="serve cases over stdin/stdout for bench.perf_ab")
    ap.add_argument("--repeats", type=int, default=2,
                    help="timed solves per case; the minimum is kept (default 2)")
    ap.add_argument("--label", default=None, help="name for this build in reports")
    ap.add_argument("--quick", action="store_true",
                    help="one instance, one seed (for local smoke tests)")
    args = ap.parse_args(argv)
    if args.worker:
        return worker(args.label, args.quick)
    if args.out is None:
        ap.error("--out is required")

    cases = suite(args.quick)

    t0 = time.time()
    results = []
    for k, (scn, spec, seed) in enumerate(cases, 1):
        results.append(run_case(scn, spec, seed, args.repeats))
        r = results[-1]
        print(f"[{k:>2}/{len(cases)}] {r['id']:<48} "
              f"{r['time_per_iter'] * 1e6:8.1f} us/iter  "
              f"{'solved' if r['solved'] else 'unsolved':>8}", flush=True)

    out = {"meta": {**meta(args.label, args.repeats), "wall_time": time.time() - t0},
           "cases": results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out} ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
