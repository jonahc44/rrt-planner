"""Benchmark harness: run every planner on every scenario, several times.

    python -m bench.run                       # default suite
    python -m bench.run --quick               # smoke test (~10 s)
    python -m bench.run --families maze --instances 10 --runs 8 --time-limit 5

Writes <out>/results.json (raw), summary.csv / summary.md, and the plots.
Requires the rrt_planner module (pip install ., or PYTHONPATH=python after a
CMake build with -DRRT_BUILD_PYTHON=ON).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import statistics
import sys
import time
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .planners import PLANNERS
from .scenarios import GENERATORS, generate


def run_one(scn, spec, run: int, time_limit: float, progress_interval: int) -> dict:
    # Deterministic across processes (str hash() is salted per process).
    seed = (scn.seed * 1_000_003 + run * 7919 + zlib.crc32(spec.name.encode())) % 2**63
    planner, problem = spec.build(scn, seed, progress_interval)
    res = planner.solve(time_limit=time_limit)  # releases the GIL
    record = {
        "family": scn.family,
        "instance": scn.instance,
        "scenario": scn.name,
        "planner": spec.name,
        "space": spec.space,
        "run": run,
        "seed": seed,
        "solved": res.solved,
        "cost": res.cost if res.solved else None,
        "first_solution_time": res.first_solution_time if res.solved else None,
        "first_solution_iteration": res.first_solution_iteration if res.solved else None,
        "iterations": res.iterations,
        "tree_size": res.tree_size,
        "elapsed": res.elapsed,
        # time, iteration, tree_size, cost (None = no solution yet)
        "convergence": [
            [round(t, 6), int(i), int(n), (round(c, 6) if math.isfinite(c) else None)]
            for t, i, n, c in res.convergence_array().tolist()
        ],
        "path": (np.round(res.dense_path(0.05)[:, :3], 4).tolist()
                 if res.solved else None),
    }
    return record


def normalise_costs(runs: list[dict]) -> None:
    """Attach a reference cost to every run.

    reference_cost = best path length any holonomic (R²) run found on the
    scenario. A Dubins path is also a valid R² path for the same disc robot,
    so this approximates a lower bound and cost_ratio shows both how far a
    run has converged and what the turning-radius constraint costs. If no R²
    planner was run, falls back to the best cost of the same planner.
    """
    best_r2: dict[str, float] = {}
    best_own: dict[tuple[str, str], float] = {}
    for r in runs:
        if not r["solved"]:
            continue
        key = (r["scenario"], r["planner"])
        best_own[key] = min(best_own.get(key, math.inf), r["cost"])
        if r["space"] == "r2":
            best_r2[r["scenario"]] = min(best_r2.get(r["scenario"], math.inf), r["cost"])
    for r in runs:
        ref = best_r2.get(r["scenario"]) or best_own.get((r["scenario"], r["planner"]))
        r["reference_cost"] = ref
        r["reference"] = "best_r2" if r["scenario"] in best_r2 else "best_same_planner"
        r["cost_ratio"] = r["cost"] / ref if r["solved"] and ref else None


def summarise(runs: list[dict], families: list[str], planners: list[str]) -> list[dict]:
    def med(xs):
        return statistics.median(xs) if xs else None

    rows = []
    for fam in families:
        for p in planners:
            rs = [r for r in runs if r["family"] == fam and r["planner"] == p]
            if not rs:
                continue
            solved = [r for r in rs if r["solved"]]
            rows.append({
                "family": fam,
                "planner": p,
                "runs": len(rs),
                "success_rate": len(solved) / len(rs),
                "median_first_solution_s": med([r["first_solution_time"] for r in solved]),
                "median_cost_ratio": med([r["cost_ratio"] for r in solved]),
                "median_final_cost": med([r["cost"] for r in solved]),
                "median_tree_size": med([r["tree_size"] for r in rs]),
                "median_iterations_per_s": med([r["iterations"] / r["elapsed"]
                                                for r in rs if r["elapsed"] > 0]),
            })
    return rows


def _fmt(v, spec):
    return "–" if v is None else format(v, spec)


def write_summary(rows: list[dict], out: Path) -> str:
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    lines = [
        "| Scenario | Planner | Success | First solution (median s) | "
        "Final cost / best R² cost (median) | Nodes (median) | Iterations/s |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['family']} | {r['planner']} | {r['success_rate']:.0%} | "
            f"{_fmt(r['median_first_solution_s'], '.3f')} | "
            f"{_fmt(r['median_cost_ratio'], '.3f')} | "
            f"{_fmt(r['median_tree_size'], ',.0f')} | "
            f"{_fmt(r['median_iterations_per_s'], ',.0f')} |")
    md = "\n".join(lines) + "\n"
    (out / "summary.md").write_text(md)
    return md


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--families", nargs="+", default=list(GENERATORS),
                    choices=list(GENERATORS))
    ap.add_argument("--planners", nargs="+", default=list(PLANNERS),
                    choices=list(PLANNERS), metavar="NAME",
                    help=f"subset of: {', '.join(PLANNERS)}")
    ap.add_argument("--instances", type=int, default=5,
                    help="scenarios per family (default 5)")
    ap.add_argument("--runs", type=int, default=4,
                    help="seeds per (scenario, planner) (default 4)")
    ap.add_argument("--time-limit", type=float, default=2.0,
                    help="seconds per run (default 2)")
    ap.add_argument("--progress-interval", type=int, default=100,
                    help="record tree size every N iterations (default 100)")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2),
                    help="parallel runs (default: half the logical CPUs). "
                         "More jobs = faster suite but noisier timings.")
    ap.add_argument("--seed", type=int, default=0, help="scenario base seed")
    ap.add_argument("--out", type=Path, default=Path("bench_results"))
    ap.add_argument("--quick", action="store_true",
                    help="2 instances x 2 runs x 0.5 s")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args(argv)
    if args.quick:
        args.instances, args.runs, args.time_limit = 2, 2, 0.5

    args.out.mkdir(parents=True, exist_ok=True)
    scenarios = [generate(f, i, args.seed) for f in args.families
                 for i in range(args.instances)]
    specs = [PLANNERS[p] for p in args.planners]
    tasks = [(s, p, r) for s in scenarios for p in specs for r in range(args.runs)]
    est = len(tasks) * args.time_limit / args.jobs
    print(f"{len(scenarios)} scenarios x {len(specs)} planners x {args.runs} runs "
          f"= {len(tasks)} runs, {args.time_limit:g} s each, {args.jobs} jobs "
          f"(~{est:.0f} s)", flush=True)

    t0 = time.time()
    runs = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futs = [pool.submit(run_one, s, p, r, args.time_limit, args.progress_interval)
                for s, p, r in tasks]
        for k, fut in enumerate(as_completed(futs), 1):
            runs.append(fut.result())
            if k % max(1, len(tasks) // 20) == 0 or k == len(tasks):
                print(f"  {k}/{len(tasks)} done ({time.time() - t0:.0f} s)", flush=True)
    runs.sort(key=lambda r: (r["family"], r["instance"], r["planner"], r["run"]))
    normalise_costs(runs)

    meta = {
        "time_limit": args.time_limit, "instances": args.instances,
        "runs": args.runs, "jobs": args.jobs, "seed": args.seed,
        "progress_interval": args.progress_interval,
        "families": args.families, "planners": args.planners,
        "python": sys.version.split()[0], "platform": platform.platform(),
        "cpu_count": os.cpu_count(), "wall_time": time.time() - t0,
    }
    with open(args.out / "results.json", "w") as f:
        json.dump({"meta": meta, "scenarios": [s.to_dict() for s in scenarios],
                   "runs": runs}, f)

    md = write_summary(summarise(runs, args.families, args.planners), args.out)
    print()
    print(md)

    if not args.no_plots:
        from .plots import plot_all
        for p in plot_all(args.out / "results.json", args.out):
            print(f"wrote {p}")
    print(f"results in {args.out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
