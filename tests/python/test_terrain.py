"""Terrain cost maps through the bindings, on randomised terrain.

Random cases derive from a base seed (env RRT_TEST_SEED, default fixed) so a
failure can be replayed; the case seed is part of each test id / message.
"""

import collections
import math
import os

import numpy as np
import pytest

import rrt_planner as rp

BASE_SEED = int(os.environ.get("RRT_TEST_SEED", "20261024"))
SITES = list(rp.SiteType)
TERRAINS = list(rp.TerrainType)
TRAVERSABLE = [t for t in TERRAINS if rp.default_terrain_properties(t).traversable]


def random_terrain(rng, shape, choices=TERRAINS):
    """(h, w) array of terrain values drawn uniformly from `choices`."""
    return rng.choice(np.array([int(t) for t in choices]), size=shape)


def connected(terrain, a, b):
    """8-connected flood fill over traversable cells (no corner cutting)."""
    free = np.isfinite(terrain.cost_array())
    res = terrain.resolution
    h, w = free.shape
    (ac, ar), (bc, br) = [(int(p[0] / res), int(p[1] / res)) for p in (a, b)]
    if not (free[ar, ac] and free[br, bc]):
        return False
    seen = np.zeros_like(free)
    seen[ar, ac] = True
    q = collections.deque([(ar, ac)])
    while q:
        r, c = q.popleft()
        if (r, c) == (br, bc):
            return True
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if not (0 <= nr < h and 0 <= nc < w) or seen[nr, nc] or not free[nr, nc]:
                    continue
                if dr and dc and not (free[r, nc] and free[nr, c]):
                    continue
                seen[nr, nc] = True
                q.append((nr, nc))
    return False


def test_enums_and_default_table():
    assert len(TERRAINS) == 17 and len(SITES) == 5
    assert rp.terrain_name(rp.TerrainType.WET_CLAY) == "wet_clay"
    assert rp.site_name(rp.SiteType.ROAD_WORKS) == "road_works"
    assert rp.default_terrain_properties(rp.TerrainType.ASPHALT).cost == 1.0
    for t in (rp.TerrainType.EXCAVATION, rp.TerrainType.STOCKPILE,
              rp.TerrainType.FRESH_CONCRETE, rp.TerrainType.STRUCTURE):
        assert not rp.default_terrain_properties(t).traversable
    assert all(rp.default_terrain_properties(t).cost >= 1.0 for t in TRAVERSABLE)


@pytest.mark.parametrize("case", range(10))
def test_array_round_trip_and_lookups(case):
    rng = np.random.default_rng(BASE_SEED + case)
    shape = tuple(rng.integers(3, 40, size=2))
    res = float(rng.uniform(0.05, 1.0))
    origin = tuple(rng.uniform(-5, 5, size=2))
    values = random_terrain(rng, shape)
    tm = rp.TerrainMap.from_array(values, res, *origin)
    np.testing.assert_array_equal(tm.to_array(), values)
    assert (tm.height, tm.width) == shape and tm.origin == pytest.approx(origin)

    table = {int(t): tm.properties(t) for t in TERRAINS}
    expected = np.vectorize(
        lambda v: table[v].cost if table[v].traversable else math.inf)(values)
    np.testing.assert_array_equal(tm.cost_array(), expected)

    # Point lookups at random cell centres agree with the arrays.
    for _ in range(50):
        r, c = rng.integers(shape[0]), rng.integers(shape[1])
        x, y = origin[0] + (c + 0.5) * res, origin[1] + (r + 0.5) * res
        assert int(tm.terrain_at(x, y)) == values[r, c]
        assert tm.cost_at(x, y) == expected[r, c]
        assert tm.traversable_at(x, y) == math.isfinite(expected[r, c])

    occ = tm.occupancy_grid().to_array()
    np.testing.assert_array_equal(occ, np.isinf(expected).astype(float))
    norm = tm.normalized_cost_grid().to_array()
    finite = np.isfinite(expected)
    if finite.any():
        np.testing.assert_allclose(norm[finite], expected[finite] / expected[finite].max())
    assert np.all(norm[~finite] == 1.0)


def test_from_array_rejects_bad_values():
    with pytest.raises(ValueError):
        rp.TerrainMap.from_array(np.array([[0, 17]]), 1.0)
    with pytest.raises(ValueError):
        rp.TerrainMap.from_array(np.array([[-1]]), 1.0)


def test_properties_override_changes_the_cost():
    tm = rp.TerrainMap(4, 4, 1.0, fill=rp.TerrainType.MUD)
    path = rp.StraightLinePath([0.5, 0.5], [3.5, 3.5])
    before = rp.TerrainCost(tm).motion_cost(path)
    assert before == pytest.approx(5.0 * path.length())
    tm.set_properties(rp.TerrainType.MUD, rp.TerrainProperties(cost=2.0))
    assert rp.TerrainCost(tm).motion_cost(path) == pytest.approx(2.0 * path.length())
    with pytest.raises(ValueError):
        tm.set_properties(rp.TerrainType.MUD, rp.TerrainProperties(cost=-1.0))


@pytest.mark.parametrize("case", range(10))
def test_terrain_cost_matches_a_fine_numpy_integral(case):
    rng = np.random.default_rng(BASE_SEED * 7 + case)
    values = random_terrain(rng, (20, 20), TRAVERSABLE)
    tm = rp.TerrainMap.from_array(values, 0.5)
    cost_grid = tm.cost_array()
    cost = rp.TerrainCost(tm, integration_step=0.01)
    for _ in range(10):
        a, b = rng.uniform(0, 10, size=2), rng.uniform(0, 10, size=2)
        n = 20000
        t = (np.arange(n) + 0.5) / n
        pts = a + t[:, None] * (b - a)
        idx = np.minimum((pts / 0.5).astype(int), 19)
        ref = cost_grid[idx[:, 1], idx[:, 0]].sum() * np.linalg.norm(b - a) / n
        got = cost.motion_cost(rp.StraightLinePath(list(a), list(b)))
        # Each of <= 42 cell crossings can misattribute up to one step.
        assert got == pytest.approx(ref, abs=42 * 0.01 * 6.0)


@pytest.mark.parametrize("site", SITES, ids=lambda s: rp.site_name(s))
def test_random_sites_are_seeded_and_keep_points_clear(site):
    rng = np.random.default_rng(BASE_SEED)
    for _ in range(10):
        seed = int(rng.integers(2**63))
        pts = [list(p) for p in rng.uniform(0, 10, size=(3, 2))]
        a = rp.TerrainMap.random_site(site, 50, 50, 0.2, seed, pts, 0.6)
        b = rp.TerrainMap.random_site(site, 50, 50, 0.2, seed, pts, 0.6)
        np.testing.assert_array_equal(a.to_array(), b.to_array(), err_msg=f"seed {seed}")
        for p in pts:
            for ang in np.linspace(0, 2 * np.pi, 16, endpoint=False):
                for rad in (0.0, 0.3, 0.6):
                    x, y = p[0] + rad * np.cos(ang), p[1] + rad * np.sin(ang)
                    if 0 <= x < 10 and 0 <= y < 10:
                        assert a.traversable_at(x, y), f"seed {seed} at {(x, y)}"
        # A mix of surfaces, and most of the site is drivable.
        arr = a.to_array()
        assert len(np.unique(arr)) >= 4, f"seed {seed}"
        assert np.isfinite(a.cost_array()).mean() > 0.65, f"seed {seed}"


@pytest.mark.parametrize("site", SITES, ids=lambda s: rp.site_name(s))
def test_planner_routes_around_expensive_ground(site):
    """RRT* with TerrainCost on random sites: valid paths, cost that
    matches the terrain integral, and cheaper ground than the
    length-optimal path (in aggregate over seeds)."""
    start, goal = [0.5, 0.5], [9.5, 9.5]
    rng = np.random.default_rng(BASE_SEED + 101 * int(site.value))
    sums = np.zeros(2)
    planned = 0
    while planned < 3:
        seed = int(rng.integers(2**31))
        tm = rp.TerrainMap.random_site(site, 50, 50, 0.2, seed, [start, goal], 0.5)
        if not connected(tm, start, goal):
            continue
        space = rp.RealVector2DSpace(0, 10, 0, 10)
        checker = rp.TerrainCollisionChecker(tm, 0.05)
        terrain_cost = rp.TerrainCost(tm, 0.05)

        def plan(cost):
            planner = rp.RRTStar(rp.RRTStarConfig(seed=seed))
            planner.set_problem(rp.PlanningProblem(
                space=space, checker=checker, cost=cost, start=start,
                goal=rp.GoalState(space, goal, 0.2)))
            return planner.solve(max_iterations=5000)

        by_terrain = plan(terrain_cost)
        by_length = plan(rp.PathLengthCost())
        assert by_terrain.solved and by_length.solved, f"seed {seed}"
        assert all(checker.is_motion_valid(s) for s in by_terrain.segments)
        recomputed = sum(terrain_cost.motion_cost(s) for s in by_terrain.segments)
        assert by_terrain.cost == pytest.approx(recomputed, rel=1e-9)
        length_terrain = sum(terrain_cost.motion_cost(s) for s in by_length.segments)
        assert by_terrain.cost <= 1.15 * length_terrain, f"seed {seed}"
        sums += (by_terrain.cost, length_terrain)
        planned += 1
    assert sums[0] < sums[1]
