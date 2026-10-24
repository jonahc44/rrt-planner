"""Randomised planning scenarios: narrow passages, cluttered fields, mazes.

Every scenario lives in a 10 x 10 world and is checked for feasibility for a
holonomic disc robot (grid flood fill) before it is returned. Feasibility for
a car with a turning radius is *not* guaranteed: a Dubins planner's success
rate mixes planner failures with instances it cannot solve at all.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass, field

import numpy as np

import rrt_planner as rp

WORLD = (0.0, 10.0, 0.0, 10.0)
ROBOT_RADIUS = 0.1


@dataclass
class Scenario:
    family: str
    instance: int
    seed: int
    start: tuple[float, float, float]  # x, y, heading
    goal: tuple[float, float, float]
    circles: list[tuple[float, float, float]] = field(default_factory=list)
    boxes: list[tuple[float, float, float, float]] = field(default_factory=list)
    bounds: tuple[float, float, float, float] = WORLD
    robot_radius: float = ROBOT_RADIUS

    @property
    def name(self) -> str:
        return f"{self.family}-{self.instance}"

    def make_world(self, resolution: float = 0.02) -> rp.GeometricWorld2D:
        w = rp.GeometricWorld2D(*self.bounds, robot_radius=self.robot_radius,
                                resolution=resolution)
        for c in self.circles:
            w.add_circle(*c)
        for b in self.boxes:
            w.add_box(*b)
        return w

    def to_dict(self) -> dict:
        d = asdict(self)
        d["name"] = self.name
        return d


# ---- feasibility -------------------------------------------------------------

def free_space_grid(scn: Scenario, res: float = 0.05) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Boolean (ny, nx) grid of collision-free cell centres."""
    x0, x1, y0, y1 = scn.bounds
    xs = np.arange(x0 + res / 2, x1, res)
    ys = np.arange(y0 + res / 2, y1, res)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    free = scn.make_world().are_states_valid(pts).reshape(gy.shape)
    return free, xs, ys


def is_feasible(scn: Scenario, res: float = 0.05) -> bool:
    """8-connected flood fill from start to goal over the free-space grid."""
    free, xs, ys = free_space_grid(scn, res)

    def cell(p):
        return (int(np.clip(np.searchsorted(ys, p[1]) - 1, 0, len(ys) - 1)),
                int(np.clip(np.searchsorted(xs, p[0]) - 1, 0, len(xs) - 1)))

    s, g = cell(scn.start), cell(scn.goal)
    if not (free[s] and free[g]):
        return False
    seen = np.zeros_like(free)
    seen[s] = True
    q = deque([s])
    ny, nx = free.shape
    while q:
        r, c = q.popleft()
        if (r, c) == g:
            return True
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < ny and 0 <= cc < nx and free[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    q.append((rr, cc))
    return False


# ---- generators --------------------------------------------------------------

def narrow_passage(rng: np.random.Generator, instance: int, seed: int) -> Scenario:
    """One to three full-height walls, each pierced by a single narrow gap."""
    n_walls = int(rng.integers(1, 4))
    thickness = 0.3
    xs = np.linspace(0, 10, n_walls + 2)[1:-1] + rng.uniform(-0.4, 0.4, n_walls)
    boxes = []
    for x in xs:
        gap = rng.uniform(0.45, 0.8)  # robot diameter is 0.2
        yc = rng.uniform(1.5, 8.5)
        boxes.append((x - thickness / 2, 0.0, x + thickness / 2, yc - gap / 2))
        boxes.append((x - thickness / 2, yc + gap / 2, x + thickness / 2, 10.0))
    start = (0.8, float(rng.uniform(1, 9)), 0.0)
    goal = (9.2, float(rng.uniform(1, 9)), 0.0)
    return Scenario("narrow_passage", instance, seed, start, goal, boxes=boxes)


def cluttered(rng: np.random.Generator, instance: int, seed: int,
              coverage: float = 0.25) -> Scenario:
    """Random circles and boxes until roughly `coverage` of the area is filled,
    keeping a clear disc around the start and goal."""
    start = (float(rng.uniform(0.6, 2.0)), float(rng.uniform(0.6, 2.0)),
             float(rng.uniform(-math.pi, math.pi)))
    goal = (float(rng.uniform(8.0, 9.4)), float(rng.uniform(8.0, 9.4)),
            float(rng.uniform(-math.pi, math.pi)))
    keep_clear = 1.0
    res = 0.1
    gx, gy = np.meshgrid(np.arange(res / 2, 10, res), np.arange(res / 2, 10, res))
    filled = np.zeros_like(gx, dtype=bool)
    circles, boxes = [], []
    for _ in range(500):
        if filled.mean() >= coverage:
            break
        cx, cy = rng.uniform(0, 10, 2)
        if rng.random() < 0.6:
            r = rng.uniform(0.25, 0.7)
            if min(math.dist((cx, cy), start[:2]), math.dist((cx, cy), goal[:2])) < r + keep_clear:
                continue
            circles.append((float(cx), float(cy), float(r)))
            filled |= (gx - cx) ** 2 + (gy - cy) ** 2 <= r * r
        else:
            w, h = rng.uniform(0.4, 1.2, 2)
            b = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
            if any(_box_disc_dist(b, p) < keep_clear for p in (start, goal)):
                continue
            boxes.append(tuple(float(v) for v in b))
            filled |= (gx >= b[0]) & (gx <= b[2]) & (gy >= b[1]) & (gy <= b[3])
    return Scenario("cluttered", instance, seed, start, goal, circles=circles, boxes=boxes)


def maze(rng: np.random.Generator, instance: int, seed: int, n: int = 7,
         braid: float = 0.1) -> Scenario:
    """n x n cell maze carved by randomised DFS, with a fraction of the
    remaining interior walls knocked out to create loops (so there is more
    than one homotopy class for the optimiser to choose between)."""
    cell = 10.0 / n
    t = 0.15
    # walls_v[i, j]: wall between cell (i, j) and (i + 1, j); walls_h likewise in y.
    walls_v = np.ones((n - 1, n), dtype=bool)
    walls_h = np.ones((n, n - 1), dtype=bool)
    seen = np.zeros((n, n), dtype=bool)
    stack = [(0, 0)]
    seen[0, 0] = True
    while stack:
        i, j = stack[-1]
        nbrs = [(i + di, j + dj) for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1))
                if 0 <= i + di < n and 0 <= j + dj < n and not seen[i + di, j + dj]]
        if not nbrs:
            stack.pop()
            continue
        ni, nj = nbrs[rng.integers(len(nbrs))]
        if ni != i:
            walls_v[min(i, ni), j] = False
        else:
            walls_h[i, min(j, nj)] = False
        seen[ni, nj] = True
        stack.append((ni, nj))
    for walls in (walls_v, walls_h):
        idx = np.argwhere(walls)
        knock = rng.random(len(idx)) < braid
        walls[tuple(idx[knock].T)] = False

    boxes = []
    for i, j in np.argwhere(walls_v):
        x = (i + 1) * cell
        boxes.append((x - t / 2, j * cell - t / 2, x + t / 2, (j + 1) * cell + t / 2))
    for i, j in np.argwhere(walls_h):
        y = (j + 1) * cell
        boxes.append((i * cell - t / 2, y - t / 2, (i + 1) * cell + t / 2, y + t / 2))
    boxes = [tuple(float(v) for v in b) for b in boxes]
    headings = (0.0, math.pi / 2, math.pi, -math.pi / 2)
    start = (cell / 2, cell / 2, float(headings[rng.integers(4)]))
    goal = (10 - cell / 2, 10 - cell / 2, float(headings[rng.integers(4)]))
    return Scenario("maze", instance, seed, start, goal, boxes=boxes)


def _box_disc_dist(b, p) -> float:
    dx = p[0] - min(max(p[0], b[0]), b[2])
    dy = p[1] - min(max(p[1], b[1]), b[3])
    return math.hypot(dx, dy)


GENERATORS = {
    "narrow_passage": narrow_passage,
    "cluttered": cluttered,
    "maze": maze,
}


def generate(family: str, instance: int, base_seed: int = 0,
             max_attempts: int = 100) -> Scenario:
    """Deterministic scenario for (family, instance, base_seed); resamples
    until the instance is feasible for the disc robot."""
    gen = GENERATORS[family]
    family_id = sorted(GENERATORS).index(family)
    for attempt in range(max_attempts):
        seed = int(np.random.SeedSequence([base_seed, family_id, instance, attempt])
                   .generate_state(1)[0])
        scn = gen(np.random.default_rng(seed), instance, seed)
        if is_feasible(scn):
            return scn
    raise RuntimeError(f"could not generate a feasible {family} scenario")
