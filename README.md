# rrt-planner

Sampling-based path planning in C++17:

- **RRT\*** in continuous 2D (holonomic point robot).
- **Kinodynamic RRT\*** for a car with a minimum turning radius. Every tree
  edge is an exact Dubins curve, so paths are curvature-bounded and
  heading-continuous.

Both use **the same planner class**. Only the `StateSpace` differs.

## Build

```sh
cmake -S . -B build -G Ninja
cmake --build build
ctest --test-dir build          # or ./build/rrt_tests
```

Dependencies: [nanoflann](https://github.com/jlblancoc/nanoflann) (KD-tree)
and GoogleTest (tests only). CMake uses installed copies when it finds them
and otherwise fetches them with `FetchContent`.

## Examples

```sh
./build/plan_r2     [iterations] [out.svg]
./build/plan_dubins [iterations] [turning_radius] [out.svg]
```

Each example writes an SVG showing the obstacles, the search tree and the
solution.

## Architecture

```
include/rrt/
  core/         abstract interfaces + planner-facing types
    state_space.hpp        StateSpace        sample / distance / steer / embed
    local_path.hpp         LocalPath         an edge, parameterised by arc length
    collision_checker.hpp  CollisionChecker  isStateValid / isMotionValid
    cost_function.hpp      CostFunction      motionCost / combine (+ PathLengthCost)
    goal.hpp               Goal              isSatisfied / sample (+ GoalState)
    planner.hpp            Planner, PlanningProblem, Termination, PlannerResult
    nearest_neighbors.hpp  NearestNeighbors (KD-tree via nanoflann, brute force)
  spaces/       RealVector2DSpace, DubinsStateSpace, dubins:: closed-form solver
  collision/    GeometricWorld2D (circles/boxes, disc robot), OccupancyGrid + GridCollisionChecker
  cost/         CostmapCost (∫ 1 + w·c(x,y) ds over a grid)
  planners/     RRTStar
```

`RRTStar` only ever calls the abstract interfaces. A `PlanningProblem` is just
four `shared_ptr`s plus a start state:

```cpp
PlanningProblem problem;
problem.space   = std::make_shared<DubinsStateSpace>(/*rho=*/0.6, 0, 10, 0, 10);
problem.checker = world;                                  // any CollisionChecker
problem.cost    = std::make_shared<PathLengthCost>();     // or CostmapCost, ...
problem.start   = {1.0, 1.0, 0.0};
problem.goal    = std::make_shared<GoalState>(problem.space, State{9, 9, -M_PI/2}, 0.05);

RRTStarConfig config;
config.max_step = 1.0;
config.max_rewire_radius = 3.0;
RRTStar planner(config);
planner.setProblem(problem);
PlannerResult r = planner.solve({/*max_iterations=*/5000});
auto dense = r.densePath(0.05);
```

### Adding a vehicle model

Subclass `StateSpace` and implement:

| method | meaning |
|---|---|
| `sampleUniform` | uniform state in bounds |
| `distance(from, to)` | optimal steering cost; **may be asymmetric** |
| `connect(from, to)` | exact steering as a `LocalPath` (return `nullptr` if impossible) |
| `embed`, `embeddingDimension` | Euclidean embedding with ‖embed(a) − embed(b)‖ ≤ distance(a, b) |
| `measure` | volume of the space (sizes the RRT\* rewiring ball) |

`steer()` has a default that truncates `connect()`. Override it if your model
can only steer approximately (for example by forward-simulating controls).

The embedding's lower-bound property is what keeps nearest-neighbour queries
**exact** for non-Euclidean metrics. The KD-tree searches the embedding with a
conservative radius and then filters candidates with the true `distance()`. For
Dubins the embedding is `(x, y)`, since a Dubins path is never shorter than the
straight line between its endpoints. The test suite checks the KD-tree against
brute force for both spaces.

### Adding an environment or objective

- **Environment:** subclass `CollisionChecker` and implement
  `isStateValid`. `isMotionValid` defaults to sampling the edge every
  `resolution()` units. Override it if you have something better, such as a
  swept-volume test.
- **Objective:** subclass `CostFunction` and implement `motionCost(path)`.
  Costs must be non-negative and additive under `combine`. The tests include a
  cost-map objective that steers the path around an expensive region.

Workspace checkers and cost maps read position from `state[0], state[1]`.
Both bundled spaces follow that layout.

## RRT\* details

- Rewiring radius: `r(n) = min(r_max, γ (log n / n)^(1/d))`, with
  `γ = rewire_factor · γ*` from Karaman & Frazzoli (2011), clamped below by
  `min_rewire_radius`.
- Choose-parent sorts candidates by cost and collision-checks them lazily,
  best first.
- Directional neighbourhoods for asymmetric metrics: parents come from
  `nearTo(x_new)` (nodes that can reach x_new) and rewiring targets come from
  `nearFrom(x_new)` (nodes reachable from x_new).
- When a node is rewired, the cost change is pushed down its subtree.
- `solve()` can be called repeatedly. Each call keeps refining the same tree.

## Tests

- Dubins: every feasible word lands exactly on the target pose, the shortest
  path is never shorter than the straight line, and the curvature bound holds
  along sampled paths.
- Nearest neighbours: the nanoflann KD-tree matches brute force exactly
  (nearest, `nearTo`, `nearFrom`) in R² and for Dubins.
- RRT\*: converges to within 3% of the straight-line optimum in free space,
  keeps tree invariants (costs, edge endpoints), never increases cost across
  `solve()` calls, produces collision-free and curvature-bounded Dubins paths,
  and works with a cost-map objective.
