"""Planner configurations under benchmark.

Each spec turns a Scenario into a ready-to-solve (planner, problem) pair.
Adding a configuration is one entry in PLANNERS.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import rrt_planner as rp

from .scenarios import Scenario


@dataclass(frozen=True)
class PlannerSpec:
    name: str
    space: str  # "r2" or "dubins"
    turning_radius: float = 0.0
    goal_tolerance: float = 0.1
    config: dict = field(default_factory=dict)

    def build(self, scn: Scenario, seed: int, progress_interval: int = 0):
        x0, x1, y0, y1 = scn.bounds
        if self.space == "r2":
            space = rp.RealVector2DSpace(x0, x1, y0, y1)
            start, goal = list(scn.start[:2]), list(scn.goal[:2])
        elif self.space == "dubins":
            space = rp.DubinsStateSpace(self.turning_radius, x0, x1, y0, y1)
            start, goal = list(scn.start), list(scn.goal)
        else:
            raise ValueError(f"unknown space {self.space!r}")
        problem = rp.PlanningProblem(
            space=space,
            checker=scn.make_world(),
            cost=rp.PathLengthCost(),
            start=start,
            goal=rp.GoalState(space, goal, self.goal_tolerance),
        )
        planner = rp.RRTStar(rp.RRTStarConfig(
            seed=seed, progress_interval=progress_interval, **self.config))
        planner.set_problem(problem)
        return planner, problem


PLANNERS = {
    spec.name: spec
    for spec in [
        PlannerSpec("RRT* (R²)", "r2", goal_tolerance=0.1,
                    config=dict(max_step=0.5, max_rewire_radius=1.5)),
        PlannerSpec("RRT* Dubins ρ=0.5", "dubins", turning_radius=0.5,
                    goal_tolerance=0.05,
                    config=dict(max_step=1.0, max_rewire_radius=3.0)),
        PlannerSpec("RRT* Dubins ρ=1.0", "dubins", turning_radius=1.0,
                    goal_tolerance=0.05,
                    config=dict(max_step=1.5, max_rewire_radius=4.0)),
    ]
}
