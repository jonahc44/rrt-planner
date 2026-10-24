"""Tests for the pybind11 bindings, including Python-implemented plug-ins."""

import gc
import math
import threading

import numpy as np
import pytest

import rrt_planner as rp


def wall_world():
    w = rp.GeometricWorld2D(0, 10, 0, 10, robot_radius=0.1, resolution=0.02)
    w.add_box(4.5, 0.0, 5.5, 7.0)
    return w


def r2_problem(checker=None, cost=None, space=None):
    space = space or rp.RealVector2DSpace(0, 10, 0, 10)
    return rp.PlanningProblem(
        space=space,
        checker=checker or wall_world(),
        cost=cost or rp.PathLengthCost(),
        start=[1.0, 1.0],
        goal=rp.GoalState(space, [9.0, 9.0], 0.1),
    )


def test_r2_plan_returns_numpy_outputs():
    planner = rp.RRTStar(rp.RRTStarConfig(max_step=0.5, seed=3))
    planner.set_problem(r2_problem())
    res = planner.solve(max_iterations=2000)
    assert res.solved
    assert res.status == rp.PlannerStatus.SOLVED
    wp = res.waypoints
    assert wp.shape[1] == 2 and wp.shape[0] == len(res.segments) + 1
    np.testing.assert_allclose(wp[0], [1, 1])
    dense = res.dense_path(0.05)
    assert np.all(np.linalg.norm(np.diff(dense, axis=0), axis=1) <= 0.05 + 1e-9)
    conv = res.convergence_array()
    assert conv.shape[1] == 4
    assert np.all(np.diff(conv[:, 3]) <= 0)
    assert conv[-1, 3] == pytest.approx(res.cost)
    assert planner.tree_states().shape == (res.tree_size, 2)
    assert planner.tree_parents()[0] == -1


def test_termination_object_and_kwargs_agree():
    a, b = rp.RRTStar(), rp.RRTStar()
    a.set_problem(r2_problem())
    b.set_problem(r2_problem())
    ra = a.solve(rp.Termination(max_iterations=500))
    rb = b.solve(max_iterations=500)
    assert ra.cost == rb.cost and ra.tree_size == rb.tree_size


def test_dubins_plan_and_module():
    rho = 0.6
    space = rp.DubinsStateSpace(rho, 0, 10, 0, 10)
    problem = rp.PlanningProblem(
        space=space, checker=wall_world(), cost=rp.PathLengthCost(),
        start=[1, 1, 0], goal=rp.GoalState(space, [8, 2, math.pi / 2], 0.05))
    planner = rp.RRTStar(rp.RRTStarConfig(max_step=1.0, max_rewire_radius=3.0))
    planner.set_problem(problem)
    res = planner.solve(max_iterations=2000)
    assert res.solved
    assert isinstance(res.segments[0], rp.DubinsLocalPath)
    p = rp.dubins.shortest_path(rp.dubins.Pose(0, 0, 0), rp.dubins.Pose(3, 0, 0), 1.0)
    assert p.length() == pytest.approx(3.0)
    assert p.word == rp.dubins.Word.LSL or p.params[0] == pytest.approx(0)


def test_vectorised_validity():
    w = wall_world()
    pts = np.array([[1.0, 1.0], [5.0, 3.0], [5.0, 9.0]])
    assert list(w.are_states_valid(pts)) == [True, False, True]


# ---- Python plug-ins ---------------------------------------------------------

class PyLine(rp.LocalPath):
    def __init__(self, a, b):
        super().__init__()
        self.a, self.b = np.asarray(a, float), np.asarray(b, float)
        self.len = float(np.linalg.norm(self.b - self.a))

    def length(self):
        return self.len

    def interpolate(self, s):
        t = 0.0 if self.len == 0 else min(max(s / self.len, 0.0), 1.0)
        return list(self.a + t * (self.b - self.a))


class PyR2(rp.StateSpace):
    """R^2 implemented in Python, to show a new vehicle model plugs in."""

    def __init__(self):
        super().__init__()
        self.samples = []

    def dimension(self):
        return 2

    def sample_uniform(self, rng):
        s = [rng.uniform(0, 10), rng.uniform(0, 10)]
        self.samples.append(tuple(s))
        return s

    def distance(self, a, b):
        return math.dist(a, b)

    def connect(self, a, b):
        return PyLine(a, b)

    def embedding_dimension(self):
        return 2

    def embed(self, s):
        return [s[0], s[1]]

    def measure(self):
        return 100.0


class PyDiscWorld(rp.CollisionChecker):
    def __init__(self):
        super().__init__(resolution=0.05)

    def is_state_valid(self, s):
        return 0 <= s[0] <= 10 and 0 <= s[1] <= 10 and math.dist(s, (5, 5)) > 2


class PyLength(rp.CostFunction):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def motion_cost(self, path):
        self.calls += 1
        return path.length()


class PyBoxGoal(rp.Goal):
    def is_satisfied(self, s):
        return s[0] > 8.5 and s[1] > 8.5

    def sample(self, rng):
        return [rng.uniform(8.5, 10), rng.uniform(8.5, 10)]


def make_python_problem():
    space = PyR2()
    cost = PyLength()
    problem = rp.PlanningProblem(space=space, checker=PyDiscWorld(), cost=cost,
                                 start=[1, 1], goal=PyBoxGoal())
    return problem, space, cost


def test_python_plugins_drive_cpp_planner():
    problem, space, cost = make_python_problem()
    planner = rp.RRTStar(rp.RRTStarConfig(max_step=1.0))
    planner.set_problem(problem)
    res = planner.solve(max_iterations=800)
    assert res.solved
    assert cost.calls > 0
    # The C++ generator is passed by reference, so draws must not repeat.
    assert len(set(space.samples)) == len(space.samples) > 100
    assert all(math.dist(p, (5, 5)) > 2 for p in res.dense_path(0.05)[:, :2])
    assert isinstance(res.segments[0], PyLine)


def test_python_objects_survive_after_python_refs_dropped():
    planner = rp.RRTStar(rp.RRTStarConfig(max_step=1.0))
    problem, _, _ = make_python_problem()
    planner.set_problem(problem)
    del problem
    gc.collect()
    res = planner.solve(max_iterations=300)
    gc.collect()
    assert res.solved
    # Edges are PyLine objects, or C++ TruncatedPaths wrapping them (from the
    # default steer()); both must still be callable after a GC pass.
    assert all(isinstance(s, (PyLine, rp.TruncatedPath)) for s in res.segments)
    assert any(isinstance(s, PyLine) for s in res.segments)
    for seg, nxt in zip(res.segments, res.waypoints[1:]):
        np.testing.assert_allclose(seg.end(), nxt, atol=1e-9)


class NoConnectSpace(PyR2):
    def connect(self, a, b):
        return None


def test_none_from_connect_means_no_motion():
    planner = rp.RRTStar()
    planner.set_problem(r2_problem(space=NoConnectSpace(), checker=PyDiscWorld()))
    res = planner.solve(max_iterations=50)
    assert not res.solved and res.tree_size == 1


def test_missing_override_raises():
    class Incomplete(rp.CollisionChecker):
        pass

    planner = rp.RRTStar()
    with pytest.raises(RuntimeError, match="is_state_valid"):
        planner.set_problem(r2_problem(checker=Incomplete()))


def test_solve_releases_gil_for_threads():
    results = []

    def work(seed):
        p = rp.RRTStar(rp.RRTStarConfig(seed=seed))
        p.set_problem(r2_problem())
        results.append(p.solve(max_iterations=1500))

    threads = [threading.Thread(target=work, args=(s,)) for s in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(results) == 4 and all(r.solved for r in results)
