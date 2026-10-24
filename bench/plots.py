"""Plots for benchmark results.

    python -m bench.plots bench_results/results.json [--out DIR]

Figures (one panel per scenario family):
  success_vs_time.png  share of runs that have found a solution by time t
  convergence.png      anytime curve: median cost / reference cost vs time,
                       with the interquartile band
  tree_growth.png      median tree size vs time
  summary.png          success rate, time to first solution, final cost ratio
                       and tree size at the time limit (median + IQR)
  scenarios.png        example instances with each planner's best path

The reference cost is the best R² (holonomic) path length found on the
scenario: see bench.run.normalise_costs. Exact numbers are in summary.md /
summary.csv next to the figures.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from .planners import PLANNERS  # noqa: E402

# ---- palette -----------------------------------------------------------------
# Reference categorical order; the first three slots pass the colour-vision
# checks on every pair. Slot 3 (aqua) is under 3:1 contrast on the surface,
# so every figure carries a legend and the numbers are also in summary.md.

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
OBSTACLE = "#d6d5cf"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

FAMILY_TITLES = {"narrow_passage": "Narrow passage", "cluttered": "Cluttered field",
                 "maze": "Maze"}

# Header block, in inches from the top of the figure.
_TITLE_Y, _SUB_Y, _LEGEND_Y, _BODY_Y = 0.16, 0.44, 0.66, 1.0


def planner_color(name: str, all_names: list[str]) -> str:
    """Colour follows the planner, not its position in a filtered list."""
    order = list(PLANNERS) + [n for n in all_names if n not in PLANNERS]
    return SERIES[order.index(name) % len(SERIES)]


def _style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "font.family": "sans-serif",
        "font.size": 10, "text.color": INK, "axes.labelcolor": INK_2,
        "axes.edgecolor": AXIS, "axes.linewidth": 1.0,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "grid.linestyle": "-", "axes.axisbelow": True,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
        "lines.linewidth": 2.0, "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round", "legend.frameon": False,
        "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.titlecolor": INK,
    })


def _frac(fig, inches: float) -> float:
    return 1.0 - inches / fig.get_figheight()


def _finish(fig, title: str, subtitle: str, planners: list[str], path: Path,
            kind: str = "line") -> Path:
    """Title, subtitle and a one-row legend above the panels; then save."""
    fig.text(0.008, _frac(fig, _TITLE_Y), title, ha="left", va="center",
             fontsize=13, fontweight="bold", color=INK)
    fig.text(0.008, _frac(fig, _SUB_Y), subtitle, ha="left", va="center",
             fontsize=9.5, color=INK_2)
    handles = []
    for p in planners:
        c = planner_color(p, planners)
        if kind == "line":
            handles.append(plt.Line2D([], [], color=c, lw=2, label=p))
        elif kind == "dot":
            handles.append(plt.Line2D([], [], color=c, marker="o", ms=8, lw=0,
                                      mec=SURFACE, mew=1.5, label=p))
        else:
            handles.append(Rectangle((0, 0), 1, 1, color=c, label=p))
    fig.legend(handles=handles, loc="center left", ncol=len(planners),
               bbox_to_anchor=(0.0, _frac(fig, _LEGEND_Y)), fontsize=9.5,
               handlelength=1.6, labelcolor=INK_2, columnspacing=1.6)
    fig.tight_layout(rect=(0, 0, 1, _frac(fig, _BODY_Y)))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _ratio_label(meta_runs) -> str:
    refs = {r.get("reference") for r in meta_runs}
    return ("best R² path on the scenario" if refs == {"best_r2"}
            else "best path found by the same planner")


# ---- time series helpers -----------------------------------------------------

def _step(conv, grid, col):
    """Value of a convergence column at each grid time (last sample <= t)."""
    a = np.array([[np.inf if v is None else v for v in row] for row in conv], float)
    idx = np.searchsorted(a[:, 0], grid, side="right") - 1
    return np.where(idx >= 0, a[np.clip(idx, 0, None), col], np.nan)


def _quantiles(values, qs):
    """Quantiles that treat +inf (not yet solved) as a real, worst value.
    NaN where the quantile itself is unsolved."""
    out = np.quantile(np.asarray(values, float), qs, axis=0, method="inverted_cdf")
    return np.where(np.isfinite(out), out, np.nan)


def _time_grid(meta, runs):
    firsts = [r["first_solution_time"] for r in runs if r["solved"]]
    t_min = max(1e-4, 0.5 * min(firsts)) if firsts else 1e-3
    return np.geomspace(t_min, meta["time_limit"], 400)


def _panels(families, height=3.9):
    fig, axes = plt.subplots(1, len(families), figsize=(4.4 * len(families), height),
                             sharey=True, squeeze=False)
    return fig, axes[0]


def _select(runs, **kv):
    return [r for r in runs if all(r[k] == v for k, v in kv.items())]


# ---- figures -----------------------------------------------------------------

def plot_success(data, out: Path) -> Path:
    meta, runs = data["meta"], data["runs"]
    fams, planners = meta["families"], meta["planners"]
    grid = _time_grid(meta, runs)
    fig, axes = _panels(fams)
    for ax, fam in zip(axes, fams):
        for p in planners:
            rs = _select(runs, family=fam, planner=p)
            if not rs:
                continue
            t = np.array([r["first_solution_time"] if r["solved"] else np.inf for r in rs])
            share = (t[None, :] <= grid[:, None]).mean(axis=1) * 100
            ax.step(grid, share, where="post", color=planner_color(p, planners))
        ax.set_xscale("log")
        ax.set_ylim(-2, 102)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_title(FAMILY_TITLES.get(fam, fam))
        ax.set_xlabel("Planning time (s)")
    axes[0].set_ylabel("Runs solved (%)")
    return _finish(fig, "Success rate vs. planning time",
                   f"Share of runs with a solution by time t · {meta['instances']} "
                   f"scenarios × {meta['runs']} seeds per planner · "
                   f"{meta['time_limit']:g} s limit",
                   planners, out / "success_vs_time.png")


def plot_convergence(data, out: Path) -> Path:
    meta, runs = data["meta"], data["runs"]
    fams, planners = meta["families"], meta["planners"]
    grid = _time_grid(meta, runs)
    fig, axes = _panels(fams, height=4.1)
    hi = 1.05
    for ax, fam in zip(axes, fams):
        for p in planners:
            rs = _select(runs, family=fam, planner=p)
            if not rs:
                continue
            curves = [(_step(r["convergence"], grid, 3) / r["reference_cost"])
                      if r["reference_cost"] else np.full_like(grid, np.inf)
                      for r in rs]
            q25, q50, q75 = _quantiles(np.array(curves), [0.25, 0.5, 0.75])
            color = planner_color(p, planners)
            ax.fill_between(grid, q25, q75, color=color, alpha=0.10, lw=0)
            ax.plot(grid, q50, color=color)
            if np.isfinite(q50).any():
                # Keep every median's converged tail in view.
                hi = max(hi, float(np.nanmax(q50[len(q50) // 2:])))
        ax.set_xscale("log")
        ax.set_title(FAMILY_TITLES.get(fam, fam))
        ax.set_xlabel("Planning time (s)")
    axes[0].set_ylim(0.98, 1 + (hi - 1) * 1.6)
    axes[0].set_ylabel("Path cost / reference")
    return _finish(fig, "Anytime convergence",
                   f"Median path cost ÷ {_ratio_label(runs)} · band = interquartile "
                   f"range · a line starts once half the runs have a solution",
                   planners, out / "convergence.png")


def plot_tree_growth(data, out: Path) -> Path:
    meta, runs = data["meta"], data["runs"]
    fams, planners = meta["families"], meta["planners"]
    grid = np.geomspace(1e-3, meta["time_limit"], 300)
    fig, axes = _panels(fams)
    for ax, fam in zip(axes, fams):
        for p in planners:
            rs = _select(runs, family=fam, planner=p)
            if not rs:
                continue
            sizes = np.array([_step(r["convergence"], grid, 2) for r in rs])
            ax.plot(grid, np.nanmedian(sizes, axis=0), color=planner_color(p, planners))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.set_title(FAMILY_TITLES.get(fam, fam))
        ax.set_xlabel("Planning time (s)")
    axes[0].set_ylabel("Tree vertices (median)")
    return _finish(fig, "Tree growth",
                   "Median tree size vs. planning time · each Dubins vertex costs more "
                   "(curve solves, longer edges to collision-check)",
                   planners, out / "tree_growth.png")


def plot_summary(data, out: Path) -> Path:
    """Four horizontal panels; rows are (family, planner)."""
    meta, runs = data["meta"], data["runs"]
    fams, planners = meta["families"], meta["planners"]
    n = len(planners)
    step = 0.26  # row pitch inside a family group
    offsets = (np.arange(n) - (n - 1) / 2) * step

    fig, axes = plt.subplots(1, 4, figsize=(15, 1.1 + 0.42 * n * len(fams) + 1.0),
                             sharey=True)
    ax_succ, ax_time, ax_cost, ax_nodes = axes

    def rows():
        for g, fam in enumerate(fams):
            for k, p in enumerate(planners):
                yield g + offsets[k], fam, p, _select(runs, family=fam, planner=p)

    def dot(ax, y, vals, color, fmt):
        if not vals:
            ax.text(0.02, y, "no solution", transform=ax.get_yaxis_transform(),
                    va="center", ha="left", fontsize=8, color=MUTED)
            return
        med = float(np.median(vals))
        lo, hi = np.percentile(vals, [25, 75])
        ax.plot([lo, hi], [y, y], color=color, lw=2, alpha=0.45,
                solid_capstyle="round")
        ax.plot(med, y, "o", ms=8, color=color, mec=SURFACE, mew=2, zorder=3)
        ax.annotate(fmt(med), (max(hi, med), y), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=8, color=INK_2)

    for y, fam, p, rs in rows():
        color = planner_color(p, planners)
        if not rs:
            continue
        rate = 100.0 * np.mean([r["solved"] for r in rs])
        ax_succ.barh(y, rate, height=0.2, color=color, lw=0)
        ax_succ.annotate(f"{rate:.0f}%", (rate, y), xytext=(4, 0),
                         textcoords="offset points", va="center", fontsize=8,
                         color=INK_2)
        solved = [r for r in rs if r["solved"]]
        dot(ax_time, y, [r["first_solution_time"] for r in solved], color,
            lambda v: f"{v * 1000:.0f} ms" if v < 1 else f"{v:.2f} s")
        dot(ax_cost, y, [r["cost_ratio"] for r in solved if r["cost_ratio"]], color,
            lambda v: f"{v:.3f}")
        dot(ax_nodes, y, [r["tree_size"] for r in rs], color, lambda v: f"{v:,.0f}")

    ax_succ.set_xlim(0, 118)
    ax_succ.set_xticks([0, 25, 50, 75, 100])
    ax_time.set_xscale("log")
    ax_nodes.set_xscale("log")
    ax_nodes.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    lo, hi = ax_cost.get_xlim()
    ax_cost.set_xlim(min(0.99, lo), hi + (hi - 1) * 0.25)
    for ax, title in zip(axes, ["Success rate (%)", "Time to first solution (s)",
                                "Final cost / reference", "Tree vertices"]):
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
    for ax in axes[1:]:
        lo, hi = ax.get_xlim()
        ax.set_xlim(lo, hi * (1.8 if ax.get_xscale() == "log" else 1.0))
    ax_succ.set_yticks(range(len(fams)), [FAMILY_TITLES.get(f, f) for f in fams])
    ax_succ.tick_params(axis="y", labelcolor=INK, labelsize=10)
    ax_succ.set_ylim(len(fams) - 0.5, -0.5)
    return _finish(fig, "Benchmark summary",
                   f"Bars: share of runs solved · dots: median, line: interquartile range "
                   f"· {meta['time_limit']:g} s per run · first-solution time and cost "
                   f"over solved runs · cost reference = {_ratio_label(runs)}",
                   planners, out / "summary.png", kind="dot")


def _draw_scenario(ax, scn):
    x0, x1, y0, y1 = scn["bounds"]
    for cx, cy, r in scn["circles"]:
        ax.add_patch(Circle((cx, cy), r, color=OBSTACLE, lw=0))
    for bx0, by0, bx1, by1 in scn["boxes"]:
        ax.add_patch(Rectangle((bx0, by0), bx1 - bx0, by1 - by0, color=OBSTACLE, lw=0))
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color(AXIS)


def _pose(ax, pose, filled: bool):
    x, y, th = pose
    ax.annotate("", xy=(x + 0.7 * math.cos(th), y + 0.7 * math.sin(th)), xytext=(x, y),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.5), zorder=5)
    ax.plot(x, y, "o", ms=9, mfc=INK if filled else SURFACE, mec=INK, mew=2, zorder=6)


def plot_scenarios(data, out: Path, per_family: int = 3) -> Path:
    meta, runs = data["meta"], data["runs"]
    fams, planners = meta["families"], meta["planners"]
    scns = {s["name"]: s for s in data["scenarios"]}
    cols = min(per_family, meta["instances"])
    fig, axes = plt.subplots(len(fams), cols, figsize=(3.4 * cols, 3.5 * len(fams) + 1.0),
                             squeeze=False)
    for row, fam in enumerate(fams):
        for col in range(cols):
            ax = axes[row, col]
            name = f"{fam}-{col}"
            _draw_scenario(ax, scns[name])
            missing = []
            for p in planners:
                solved = [r for r in _select(runs, scenario=name, planner=p) if r["solved"]]
                if not solved:
                    missing.append(p)
                    continue
                pts = np.array(min(solved, key=lambda r: r["cost"])["path"])
                ax.plot(pts[:, 0], pts[:, 1], color=planner_color(p, planners),
                        lw=2, zorder=3)
            _pose(ax, scns[name]["start"], filled=True)
            _pose(ax, scns[name]["goal"], filled=False)
            ax.set_title(f"{FAMILY_TITLES.get(fam, fam)} · #{col}", fontsize=10)
            if missing:
                ax.text(0.5, 0.015, "no solution: " + ", ".join(missing),
                        transform=ax.transAxes, ha="center", va="bottom",
                        fontsize=8, color=INK_2, zorder=7,
                        bbox=dict(boxstyle="round,pad=0.25", fc=SURFACE, ec=AXIS,
                                  lw=0.8))
    return _finish(fig, "Example scenarios",
                   "Best path per planner over all seeds · ● start  ○ goal  "
                   "(arrow = heading, used by Dubins)",
                   planners, out / "scenarios.png")


def plot_all(results: Path, out: Path) -> list[Path]:
    _style()
    data = json.loads(Path(results).read_text())
    out.mkdir(parents=True, exist_ok=True)
    return [plot_success(data, out), plot_convergence(data, out),
            plot_tree_growth(data, out), plot_summary(data, out),
            plot_scenarios(data, out)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", type=Path)
    ap.add_argument("--out", type=Path, default=None,
                    help="output directory (default: next to results.json)")
    args = ap.parse_args(argv)
    for p in plot_all(args.results, args.out or args.results.parent):
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
