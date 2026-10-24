"""Interleaved A/B performance measurement of two builds on one machine.

    python -m bench.perf_ab --base-path BASE/python --head-path python \\
        --out-dir perf/ [--rounds 3]

Starts one long-lived `bench.perf --worker` process per build (each with its
own PYTHONPATH, both running *this* checkout's suite) and runs every case on
both builds back to back, alternating which goes first (ABBA). Machine-wide
drift such as CPU frequency changes, thermal throttling or noisy neighbours on
a shared CI runner then hits both sides of every comparison equally. Comparing
whole suites run minutes apart can show differences of almost 2x on one
machine for identical code.

Writes base.json and head.json in bench.perf's format (time = best of
--rounds). Feed them to bench.perf_compare.

Exit status: 0 on success; 3 if the base build cannot run the suite
(e.g. it predates the Python bindings). head.json is still written then,
so CI can report head numbers without a comparison.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


class Worker:
    def __init__(self, name: str, pythonpath: str, label: str, quick: bool = False):
        root = Path(pythonpath).resolve()
        if not (root / "rrt_planner").is_dir():
            raise RuntimeError(f"{name}: no rrt_planner package in {root}")
        env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(root), str(REPO)]))
        self.name = name
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "bench.perf", "--worker", "--label", label]
            + (["--quick"] if quick else []),
            cwd=REPO, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, bufsize=1)
        line = self.proc.stdout.readline()
        if not line:
            self.proc.wait()
            raise RuntimeError(f"{name} worker failed to start "
                               f"(exit {self.proc.returncode})")
        hello = json.loads(line)
        self.meta, self.cases = hello["meta"], hello["cases"]
        # Never silently measure some other installed copy (e.g. site-packages).
        if not Path(self.meta["module"]).resolve().is_relative_to(root):
            self.close()
            raise RuntimeError(f"{name}: imported {self.meta['module']}, "
                               f"not the build in {root}")

    def run(self, case_id: str, work: bool) -> dict:
        self.proc.stdin.write(json.dumps({"id": case_id, "repeats": 1, "work": work}) + "\n")
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError(f"{self.name} worker died on {case_id}")
        return json.loads(line)

    def close(self):
        if self.proc.stdin:
            self.proc.stdin.close()
        self.proc.wait(timeout=30)


def merge(first: dict, later: dict) -> dict:
    first["time_samples"] += later["time_samples"]
    first["time_per_iter"] = min(first["time_samples"])
    return first


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-path", required=True,
                    help="directory containing the base build's rrt_planner package")
    ap.add_argument("--head-path", required=True,
                    help="directory containing the head build's rrt_planner package")
    ap.add_argument("--base-label", default="base")
    ap.add_argument("--head-label", default="head")
    ap.add_argument("--rounds", type=int, default=3,
                    help="timed solves per case per build; best is kept (default 3)")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--quick", action="store_true", help="smaller suite (smoke tests)")
    args = ap.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    head = Worker("head", args.head_path, args.head_label, args.quick)
    try:
        base = Worker("base", args.base_path, args.base_label, args.quick)
    except (RuntimeError, json.JSONDecodeError) as e:
        print(f"::warning::{e}; measuring head only", flush=True)
        base = None

    ids = [c for c in head.cases if base is None or c in set(base.cases)]
    workers = [w for w in (base, head) if w is not None]
    results: dict[str, dict[str, dict]] = {w.name: {} for w in workers}
    t0 = time.time()
    for rnd in range(args.rounds):
        for k, cid in enumerate(ids):
            order = workers if (rnd + k) % 2 == 0 else workers[::-1]
            for w in order:
                r = w.run(cid, work=(rnd == 0))
                prev = results[w.name].get(cid)
                results[w.name][cid] = r if prev is None else merge(prev, r)
        print(f"round {rnd + 1}/{args.rounds} done ({time.time() - t0:.0f} s)", flush=True)

    for w in workers:
        cases = [results[w.name][cid] for cid in ids]
        meta = {**w.meta, "repeats": args.rounds, "interleaved": base is not None,
                "wall_time": time.time() - t0}
        path = args.out_dir / f"{w.name}.json"
        path.write_text(json.dumps({"meta": meta, "cases": cases}, indent=1))
        print(f"wrote {path}")
        w.close()
    return 0 if base is not None else 3


if __name__ == "__main__":
    sys.exit(main())
