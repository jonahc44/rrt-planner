"""Compare two (sets of) bench.perf results and flag regressions.

    python -m bench.perf_compare --base base*.json --head head*.json \\
        [--report report.md] [--allow-regression]

Several files per side (e.g. interleaved base/head/base/head rounds on the
same machine) are merged by taking each case's fastest time, which rejects
transient interference from noisy neighbours.

A regression fails the comparison (exit code 1) when, for any planner group
or overall:

  time    total head time / total base time (fixed iterations, so this is
          time per iteration weighted by case cost; cheap cases cannot add
          noise) exceeds --max-time (default 1.10 overall, with 1.5x the
          headroom for a single planner group)
  work    geometric-mean head/base calls per iteration exceeds --max-work
          (default 1.08). Counters are deterministic, so this is noise-free.
  cost    geometric-mean head/base final cost on cases both solve exceeds
          --max-cost (default 1.03)
  success solved fraction drops by more than --max-success-drop (0.15)

The report is Markdown; it is also appended to $GITHUB_STEP_SUMMARY when
that is set. --allow-regression reports but always exits 0.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

WORK_KEYS = ("distance_calls", "connect_calls", "state_checks", "motion_checks",
             "checked_length")


def load(paths: list[Path]) -> tuple[dict, list[dict]]:
    """Merge result files: fastest time per case; counters and quality must
    agree across files (same code, same seeds)."""
    merged: dict[str, dict] = {}
    metas = []
    for p in paths:
        data = json.loads(Path(p).read_text())
        metas.append(data["meta"])
        for c in data["cases"]:
            prev = merged.get(c["id"])
            if prev is None:
                merged[c["id"]] = dict(c)
                continue
            prev["time_per_iter"] = min(prev["time_per_iter"], c["time_per_iter"])
            prev["time_samples"] = prev["time_samples"] + c["time_samples"]
            if prev.get("work") != c.get("work") or prev["cost"] != c["cost"]:
                print(f"warning: {c['id']} differs between runs of the same build "
                      f"(non-deterministic work?)", file=sys.stderr)
    return metas[0], list(merged.values())


def geomean(xs):
    xs = [x for x in xs if x and x > 0 and math.isfinite(x)]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None


def compare(base: list[dict], head: list[dict]):
    b = {c["id"]: c for c in base}
    h = {c["id"]: c for c in head}
    common = sorted(set(b) & set(h))
    rows = []
    for cid in common:
        cb, ch = b[cid], h[cid]
        row = {"id": cid, "planner": ch["planner"],
               "time": ch["time_per_iter"] / cb["time_per_iter"],
               "base_us": cb["time_per_iter"] * 1e6, "head_us": ch["time_per_iter"] * 1e6,
               "solved_base": cb["solved"], "solved_head": ch["solved"]}
        if "work" in cb and "work" in ch:
            ratios = []
            for k in ("distance_calls", "connect_calls", "checked_length"):
                vb, vh = cb["work"].get(k, 0), ch["work"].get(k, 0)
                if vb > 0:
                    ratios.append(vh / vb)
            row["work"] = geomean(ratios)
            row["work_detail"] = {k: (cb["work"].get(k), ch["work"].get(k)) for k in WORK_KEYS}
        if cb["solved"] and ch["solved"]:
            row["cost"] = ch["cost"] / cb["cost"]
        rows.append(row)
    missing = sorted(set(b) ^ set(h))
    return rows, missing


def summarise(rows, groups):
    out = {}
    for g in groups:
        rs = [r for r in rows if g == "all" or r["planner"] == g]
        base_t = sum(r["base_us"] for r in rs)
        out[g] = {
            "cases": len(rs),
            "time": sum(r["head_us"] for r in rs) / base_t if base_t else None,
            "work": geomean([r.get("work") for r in rs]),
            "cost": geomean([r.get("cost") for r in rs]),
            "success_base": sum(r["solved_base"] for r in rs) / len(rs) if rs else None,
            "success_head": sum(r["solved_head"] for r in rs) / len(rs) if rs else None,
        }
    return out


def check(summary, args) -> list[str]:
    failures = []
    for g, s in summary.items():
        time_limit = args.max_time if g == "all" else 1 + (args.max_time - 1) * 1.5
        if s["time"] and s["time"] > time_limit:
            failures.append(f"{g}: time/iteration ×{s['time']:.3f} (limit ×{time_limit:.2f})")
        if s["work"] and s["work"] > args.max_work:
            failures.append(f"{g}: work/iteration ×{s['work']:.3f} (limit ×{args.max_work:.2f})")
        if s["cost"] and s["cost"] > args.max_cost:
            failures.append(f"{g}: final cost ×{s['cost']:.3f} (limit ×{args.max_cost:.2f})")
        if (s["success_base"] is not None
                and s["success_base"] - s["success_head"] > args.max_success_drop):
            failures.append(f"{g}: success {s['success_base']:.0%} → {s['success_head']:.0%}")
    return failures


def _r(x):
    """Format a head/base ratio; flag moves of more than 2% either way."""
    if x is None:
        return "–"
    mark = " 🔺" if x > 1.02 else (" 🟢" if x < 0.98 else "")
    return f"×{x:.3f}{mark}"


def report(base_meta, head_meta, rows, missing, summary, failures, args) -> str:
    lines = [
        "## Performance comparison",
        "",
        f"Base `{base_meta.get('label')}` vs head `{head_meta.get('label')}` · "
        f"{len(rows)} cases · fixed iterations "
        f"({', '.join(f'{k}: {v}' for k, v in head_meta.get('iterations', {}).items())}) · "
        f"ratios are head ÷ base, lower is better",
        "",
    ]
    if failures:
        lines += ["**❌ Regression detected**", ""] + [f"- {f}" for f in failures]
        if args.allow_regression:
            lines += ["", "_Allowed by `--allow-regression` (e.g. the "
                      "`perf-regression-ok` label); not failing the build._"]
    else:
        lines += ["**✅ No regression**"]
    lines += [
        "",
        "| Planner | Cases | Time / iteration | Work / iteration | Final cost | Success (base → head) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for g, s in summary.items():
        name = "**All**" if g == "all" else g
        succ = ("–" if s["success_base"] is None else
                f"{s['success_base']:.0%} → {s['success_head']:.0%}")
        lines.append(f"| {name} | {s['cases']} | {_r(s['time'])} | {_r(s['work'])} | "
                     f"{_r(s['cost'])} | {succ} |")
    worst = sorted(rows, key=lambda r: -r["time"])[:5]
    lines += ["", "<details><summary>Slowest cases relative to base</summary>", "",
              "| Case | Base µs/iter | Head µs/iter | Time | Work | Cost |",
              "|---|---:|---:|---:|---:|---:|"]
    for r in worst:
        lines.append(f"| `{r['id']}` | {r['base_us']:.1f} | {r['head_us']:.1f} | "
                     f"{_r(r['time'])} | {_r(r.get('work'))} | {_r(r.get('cost'))} |")
    lines += ["", "</details>"]
    if missing:
        lines += ["", f"_{len(missing)} cases only in one build were skipped._"]
    if not head_meta.get("has_counters") or not base_meta.get("has_counters"):
        lines += ["", "_Work counters unavailable in one build; work not compared._"]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", type=Path, nargs="+", required=True)
    ap.add_argument("--head", type=Path, nargs="+", required=True)
    ap.add_argument("--max-time", type=float, default=1.10)
    ap.add_argument("--max-work", type=float, default=1.08)
    ap.add_argument("--max-cost", type=float, default=1.03)
    ap.add_argument("--max-success-drop", type=float, default=0.15)
    ap.add_argument("--report", type=Path, default=None)
    ap.add_argument("--allow-regression", action="store_true")
    args = ap.parse_args(argv)

    base_meta, base = load(args.base)
    head_meta, head = load(args.head)
    rows, missing = compare(base, head)
    if not rows:
        print("no common cases to compare", file=sys.stderr)
        return 2
    groups = ["all"] + sorted({r["planner"] for r in rows})
    summary = summarise(rows, groups)
    failures = check(summary, args)
    md = report(base_meta, head_meta, rows, missing, summary, failures, args)
    print(md)
    if args.report:
        args.report.write_text(md)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write(md)
    return 1 if failures and not args.allow_regression else 0


if __name__ == "__main__":
    sys.exit(main())
