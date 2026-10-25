// pybind11 bindings for rrt_planner.
//
// Every abstract interface (StateSpace, LocalPath, CollisionChecker,
// CostFunction, Goal, Planner) gets a trampoline, so Python subclasses plug
// into the C++ planner exactly like C++ ones. All classes use
// py::smart_holder so C++ can safely keep shared_ptrs to Python-derived
// objects. Planner::solve releases the GIL; overrides reacquire it.

#include <pybind11/native_enum.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

#include "rrt/collision/geometric_world_2d.hpp"
#include "rrt/collision/occupancy_grid.hpp"
#include "rrt/core/collision_checker.hpp"
#include "rrt/core/cost_function.hpp"
#include "rrt/core/goal.hpp"
#include "rrt/core/instrumentation.hpp"
#include "rrt/core/nearest_neighbors.hpp"
#include "rrt/core/planner.hpp"
#include "rrt/core/state_space.hpp"
#include "rrt/cost/costmap_cost.hpp"
#include "rrt/cost/terrain.hpp"
#include "rrt/planners/rrt_star.hpp"
#include "rrt/spaces/dubins.hpp"
#include "rrt/spaces/dubins_state_space.hpp"
#include "rrt/spaces/real_vector_2d_space.hpp"

namespace py = pybind11;
using namespace rrt;

namespace {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// N x d float64 array from a list of states.
py::array_t<double> statesToArray(const std::vector<State>& states,
                                  std::size_t dim) {
  py::array_t<double> out({states.size(), dim});
  auto m = out.mutable_unchecked<2>();
  for (std::size_t i = 0; i < states.size(); ++i) {
    for (std::size_t k = 0; k < dim; ++k) m(i, k) = states[i][k];
  }
  return out;
}

std::size_t stateDim(const std::vector<State>& states) {
  return states.empty() ? 0 : states.front().size();
}

/// Look up a Python override; throws if a pure virtual is missing.
template <typename T>
py::function requireOverride(const T* self, const char* name,
                             const char* cls) {
  py::function f = py::get_override(self, name);
  if (!f) {
    throw std::runtime_error(std::string(cls) + "." + name +
                             "() is not implemented");
  }
  return f;
}

/// Python result -> LocalPathPtr, mapping None to nullptr.
LocalPathPtr toLocalPath(const py::object& o) {
  if (o.is_none()) return nullptr;
  return o.cast<std::shared_ptr<LocalPath>>();
}

// ---------------------------------------------------------------------------
// Trampolines
// ---------------------------------------------------------------------------

class PyLocalPath : public LocalPath, public py::trampoline_self_life_support {
 public:
  double length() const override {
    PYBIND11_OVERRIDE_PURE(double, LocalPath, length, );
  }
  State interpolate(double s) const override {
    PYBIND11_OVERRIDE_PURE(State, LocalPath, interpolate, s);
  }
};

class PyStateSpace : public StateSpace,
                     public py::trampoline_self_life_support {
 public:
  std::size_t dimension() const override {
    PYBIND11_OVERRIDE_PURE(std::size_t, StateSpace, dimension, );
  }
  State sampleUniform(Rng& rng) const override {
    py::gil_scoped_acquire gil;
    // Pass the generator by reference so Python draws advance C++ state.
    py::function f = requireOverride<StateSpace>(this, "sample_uniform",
                                                 "StateSpace");
    return f(py::cast(&rng, py::return_value_policy::reference))
        .cast<State>();
  }
  double distance(const State& from, const State& to) const override {
    PYBIND11_OVERRIDE_PURE(double, StateSpace, distance, from, to);
  }
  LocalPathPtr connect(const State& from, const State& to) const override {
    py::gil_scoped_acquire gil;
    py::function f = requireOverride<StateSpace>(this, "connect", "StateSpace");
    return toLocalPath(f(from, to));
  }
  LocalPathPtr steer(const State& from, const State& toward,
                     double max_length) const override {
    {
      py::gil_scoped_acquire gil;
      if (py::function f = py::get_override(
              static_cast<const StateSpace*>(this), "steer")) {
        return toLocalPath(f(from, toward, max_length));
      }
    }
    return StateSpace::steer(from, toward, max_length);
  }
  std::size_t embeddingDimension() const override {
    PYBIND11_OVERRIDE_PURE_NAME(std::size_t, StateSpace, "embedding_dimension",
                                embeddingDimension, );
  }
  void embed(const State& s, double* out) const override {
    py::gil_scoped_acquire gil;
    py::function f = requireOverride<StateSpace>(this, "embed", "StateSpace");
    const auto v = f(s).cast<std::vector<double>>();
    if (v.size() != embeddingDimension()) {
      throw std::runtime_error(
          "StateSpace.embed() returned the wrong number of values");
    }
    std::copy(v.begin(), v.end(), out);
  }
  double measure() const override {
    PYBIND11_OVERRIDE_PURE(double, StateSpace, measure, );
  }
};

class PyCollisionChecker : public CollisionChecker,
                           public py::trampoline_self_life_support {
 public:
  using CollisionChecker::CollisionChecker;
  bool isStateValid(const State& s) const override {
    PYBIND11_OVERRIDE_PURE_NAME(bool, CollisionChecker, "is_state_valid",
                                isStateValid, s);
  }
  bool isMotionValid(const LocalPath& path) const override {
    {
      py::gil_scoped_acquire gil;
      if (py::function f = py::get_override(
              static_cast<const CollisionChecker*>(this), "is_motion_valid")) {
        return f(py::cast(&path, py::return_value_policy::reference))
            .cast<bool>();
      }
    }
    return CollisionChecker::isMotionValid(path);
  }
};

class PyCostFunction : public CostFunction,
                       public py::trampoline_self_life_support {
 public:
  double motionCost(const LocalPath& path) const override {
    py::gil_scoped_acquire gil;
    py::function f =
        requireOverride<CostFunction>(this, "motion_cost", "CostFunction");
    return f(py::cast(&path, py::return_value_policy::reference))
        .cast<double>();
  }
  double identityCost() const override {
    PYBIND11_OVERRIDE_NAME(double, CostFunction, "identity_cost",
                           identityCost, );
  }
  double combine(double a, double b) const override {
    PYBIND11_OVERRIDE(double, CostFunction, combine, a, b);
  }
};

class PyGoal : public Goal, public py::trampoline_self_life_support {
 public:
  bool isSatisfied(const State& s) const override {
    PYBIND11_OVERRIDE_PURE_NAME(bool, Goal, "is_satisfied", isSatisfied, s);
  }
  State sample(Rng& rng) const override {
    py::gil_scoped_acquire gil;
    py::function f = requireOverride<Goal>(this, "sample", "Goal");
    return f(py::cast(&rng, py::return_value_policy::reference))
        .cast<State>();
  }
};

class PyPlanner : public Planner, public py::trampoline_self_life_support {
 public:
  std::string name() const override {
    PYBIND11_OVERRIDE_PURE(std::string, Planner, name, );
  }
  void setProblem(PlanningProblem problem) override {
    PYBIND11_OVERRIDE_PURE_NAME(void, Planner, "set_problem", setProblem,
                                problem);
  }
  PlannerResult solve(const Termination& term) override {
    PYBIND11_OVERRIDE_PURE(PlannerResult, Planner, solve, term);
  }
  void clear() override { PYBIND11_OVERRIDE_PURE(void, Planner, clear, ); }
};

Termination makeTermination(std::size_t max_iterations, double time_limit,
                            bool stop_on_first_solution) {
  Termination t;
  t.max_iterations = max_iterations;
  t.time_limit_s = time_limit;
  t.stop_on_first_solution = stop_on_first_solution;
  return t;
}

}  // namespace

PYBIND11_MODULE(_core, m) {
  m.doc() = "RRT* path planning with pluggable state spaces, collision "
            "checkers and cost functions";

  // ---- Rng ----------------------------------------------------------------
  py::classh<Rng>(m, "Rng",
                  "64-bit Mersenne Twister shared with the planner. Passed "
                  "to StateSpace.sample_uniform and Goal.sample.")
      .def(py::init<std::uint64_t>(), py::arg("seed") = 42)
      .def(
          "uniform",
          [](Rng& r, double lo, double hi) {
            return std::uniform_real_distribution<double>(lo, hi)(r);
          },
          py::arg("low") = 0.0, py::arg("high") = 1.0)
      .def(
          "normal",
          [](Rng& r, double mean, double stddev) {
            return std::normal_distribution<double>(mean, stddev)(r);
          },
          py::arg("mean") = 0.0, py::arg("stddev") = 1.0)
      .def(
          "integers",
          [](Rng& r, std::int64_t lo, std::int64_t hi) {
            return std::uniform_int_distribution<std::int64_t>(lo, hi - 1)(r);
          },
          py::arg("low"), py::arg("high"), "Uniform integer in [low, high).");

  // ---- LocalPath ----------------------------------------------------------
  py::classh<LocalPath, PyLocalPath>(m, "LocalPath")
      .def(py::init<>())
      .def("length", &LocalPath::length)
      .def("interpolate", &LocalPath::interpolate, py::arg("s"))
      .def("start", &LocalPath::start)
      .def("end", &LocalPath::end)
      .def(
          "sample",
          [](const LocalPath& p, double resolution) {
            const double len = p.length();
            const auto n = static_cast<std::size_t>(
                std::max(1.0, std::ceil(len / resolution)));
            std::vector<State> pts;
            pts.reserve(n + 1);
            for (std::size_t i = 0; i <= n; ++i) {
              pts.push_back(p.interpolate(len * static_cast<double>(i) /
                                          static_cast<double>(n)));
            }
            return statesToArray(pts, stateDim(pts));
          },
          py::arg("resolution") = 0.05,
          "States along the path, at most `resolution` apart, as an "
          "N x dim array.");

  py::classh<TruncatedPath, LocalPath>(m, "TruncatedPath")
      .def(py::init([](std::shared_ptr<const LocalPath> inner, double len) {
             return TruncatedPath(std::move(inner), len);
           }),
           py::arg("inner"), py::arg("max_length"));

  py::classh<StraightLinePath, LocalPath>(m, "StraightLinePath")
      .def(py::init<State, State>(), py::arg("start"), py::arg("end"));

  py::classh<DubinsLocalPath, LocalPath>(m, "DubinsLocalPath")
      .def_property_readonly("dubins_path", &DubinsLocalPath::dubinsPath);

  // ---- StateSpace ---------------------------------------------------------
  py::classh<StateSpace, PyStateSpace>(m, "StateSpace")
      .def(py::init<>())
      .def("dimension", &StateSpace::dimension)
      .def("sample_uniform", &StateSpace::sampleUniform, py::arg("rng"))
      .def("distance", &StateSpace::distance, py::arg("a"), py::arg("b"))
      .def("connect", &StateSpace::connect, py::arg("a"), py::arg("b"))
      .def("steer", &StateSpace::steer, py::arg("a"), py::arg("toward"),
           py::arg("max_length"))
      .def("embedding_dimension", &StateSpace::embeddingDimension)
      .def(
          "embed",
          [](const StateSpace& s, const State& x) {
            std::vector<double> out(s.embeddingDimension());
            s.embed(x, out.data());
            return out;
          },
          py::arg("state"))
      .def("measure", &StateSpace::measure);

  py::classh<RealVector2DSpace, StateSpace>(m, "RealVector2DSpace")
      .def(py::init<double, double, double, double>(), py::arg("x_min"),
           py::arg("x_max"), py::arg("y_min"), py::arg("y_max"));

  py::classh<DubinsStateSpace, StateSpace>(m, "DubinsStateSpace")
      .def(py::init<double, double, double, double, double>(),
           py::arg("turning_radius"), py::arg("x_min"), py::arg("x_max"),
           py::arg("y_min"), py::arg("y_max"))
      .def_property_readonly("turning_radius",
                             &DubinsStateSpace::turningRadius);

  // ---- CollisionChecker ---------------------------------------------------
  py::classh<CollisionChecker, PyCollisionChecker>(m, "CollisionChecker")
      .def(py::init<double>(), py::arg("resolution") = 0.05)
      .def("is_state_valid", &CollisionChecker::isStateValid, py::arg("state"))
      .def(
          "is_motion_valid",
          [](const CollisionChecker& c, const LocalPath* p) {
            return c.isMotionValid(*p);
          },
          py::arg("path"))
      .def_property("resolution", &CollisionChecker::resolution,
                    &CollisionChecker::setResolution)
      .def(
          "are_states_valid",
          [](const CollisionChecker& c, py::array_t<double, py::array::c_style |
                                                                py::array::forcecast>
                                            states) {
            if (states.ndim() != 2) {
              throw std::invalid_argument("expected an N x dim array");
            }
            const auto n = static_cast<std::size_t>(states.shape(0));
            const auto d = static_cast<std::size_t>(states.shape(1));
            py::array_t<bool> out(n);
            auto in = states.unchecked<2>();
            auto res = out.mutable_unchecked<1>();
            State s(d);
            py::gil_scoped_release release;
            for (std::size_t i = 0; i < n; ++i) {
              for (std::size_t k = 0; k < d; ++k) s[k] = in(i, k);
              res(i) = c.isStateValid(s);
            }
            return out;
          },
          py::arg("states"),
          "Vectorised is_state_valid over the rows of an N x dim array.");

  py::classh<Circle>(m, "Circle")
      .def(py::init<double, double, double>(), py::arg("cx"), py::arg("cy"),
           py::arg("r"))
      .def_readwrite("cx", &Circle::cx)
      .def_readwrite("cy", &Circle::cy)
      .def_readwrite("r", &Circle::r)
      .def("__repr__", [](const Circle& c) {
        return "Circle(" + std::to_string(c.cx) + ", " + std::to_string(c.cy) +
               ", " + std::to_string(c.r) + ")";
      });

  py::classh<Box>(m, "Box")
      .def(py::init<double, double, double, double>(), py::arg("x_min"),
           py::arg("y_min"), py::arg("x_max"), py::arg("y_max"))
      .def_readwrite("x_min", &Box::x_min)
      .def_readwrite("y_min", &Box::y_min)
      .def_readwrite("x_max", &Box::x_max)
      .def_readwrite("y_max", &Box::y_max);

  py::classh<GeometricWorld2D, CollisionChecker>(m, "GeometricWorld2D")
      .def(py::init<double, double, double, double, double, double>(),
           py::arg("x_min"), py::arg("x_max"), py::arg("y_min"),
           py::arg("y_max"), py::arg("robot_radius") = 0.0,
           py::arg("resolution") = 0.05)
      .def("add_circle",
           [](GeometricWorld2D& w, double cx, double cy, double r) {
             w.addCircle({cx, cy, r});
           },
           py::arg("cx"), py::arg("cy"), py::arg("r"))
      .def("add_box",
           [](GeometricWorld2D& w, double x0, double y0, double x1, double y1) {
             w.addBox({x0, y0, x1, y1});
           },
           py::arg("x_min"), py::arg("y_min"), py::arg("x_max"),
           py::arg("y_max"))
      .def_property_readonly("circles", &GeometricWorld2D::circles)
      .def_property_readonly("boxes", &GeometricWorld2D::boxes)
      .def_property_readonly("bounds", [](const GeometricWorld2D& w) {
        return py::make_tuple(w.xMin(), w.xMax(), w.yMin(), w.yMax());
      });

  py::classh<OccupancyGrid>(m, "OccupancyGrid")
      .def(py::init<std::size_t, std::size_t, double, double, double>(),
           py::arg("width"), py::arg("height"), py::arg("resolution"),
           py::arg("origin_x") = 0.0, py::arg("origin_y") = 0.0)
      .def_static(
          "from_array",
          [](py::array_t<double, py::array::c_style | py::array::forcecast> a,
             double resolution, double ox, double oy) {
            if (a.ndim() != 2) throw std::invalid_argument("expected 2D array");
            const auto h = static_cast<std::size_t>(a.shape(0));
            const auto w = static_cast<std::size_t>(a.shape(1));
            OccupancyGrid g(w, h, resolution, ox, oy);
            auto v = a.unchecked<2>();
            for (std::size_t r = 0; r < h; ++r)
              for (std::size_t c = 0; c < w; ++c) g.at(c, r) = v(r, c);
            return g;
          },
          py::arg("values"), py::arg("resolution"), py::arg("origin_x") = 0.0,
          py::arg("origin_y") = 0.0,
          "Build from an (height, width) array; row 0 is the minimum y.")
      .def_static("from_ascii", &OccupancyGrid::fromAscii, py::arg("rows"),
                  py::arg("resolution"), py::arg("origin_x") = 0.0,
                  py::arg("origin_y") = 0.0)
      .def("to_array",
           [](const OccupancyGrid& g) {
             py::array_t<double> out({g.height(), g.width()});
             auto m = out.mutable_unchecked<2>();
             for (std::size_t r = 0; r < g.height(); ++r)
               for (std::size_t c = 0; c < g.width(); ++c) m(r, c) = g.at(c, r);
             return out;
           })
      .def("value_at", &OccupancyGrid::valueAt, py::arg("x"), py::arg("y"),
           py::arg("outside") = 1.0)
      .def_property_readonly("width", &OccupancyGrid::width)
      .def_property_readonly("height", &OccupancyGrid::height)
      .def_property_readonly("resolution", &OccupancyGrid::resolution)
      .def_property_readonly("origin", [](const OccupancyGrid& g) {
        return py::make_tuple(g.originX(), g.originY());
      });

  py::classh<GridCollisionChecker, CollisionChecker>(m, "GridCollisionChecker")
      .def(py::init<std::shared_ptr<const OccupancyGrid>, double, double>(),
           py::arg("grid"), py::arg("occupied_threshold") = 0.5,
           py::arg("resolution") = 0.05);

  // ---- Instrumentation ----------------------------------------------------
  py::classh<CountingStateSpace, StateSpace>(
      m, "CountingStateSpace",
      "Wraps a StateSpace and counts calls (thread-safe, relaxed atomics).")
      .def(py::init<std::shared_ptr<const StateSpace>>(), py::arg("inner"))
      .def_property_readonly("sample_calls", &CountingStateSpace::sampleCalls)
      .def_property_readonly("distance_calls", &CountingStateSpace::distanceCalls)
      .def_property_readonly("connect_calls", &CountingStateSpace::connectCalls)
      .def_property_readonly("steer_calls", &CountingStateSpace::steerCalls)
      .def("reset", &CountingStateSpace::reset);

  py::classh<CountingCollisionChecker, CollisionChecker>(
      m, "CountingCollisionChecker",
      "Wraps a CollisionChecker and counts calls and checked arc length.")
      .def(py::init<std::shared_ptr<const CollisionChecker>>(), py::arg("inner"))
      .def_property_readonly("state_calls", &CountingCollisionChecker::stateCalls)
      .def_property_readonly("motion_calls", &CountingCollisionChecker::motionCalls)
      .def_property_readonly("checked_length",
                             &CountingCollisionChecker::checkedLength)
      .def("reset", &CountingCollisionChecker::reset);

  // ---- CostFunction -------------------------------------------------------
  py::classh<CostFunction, PyCostFunction>(m, "CostFunction")
      .def(py::init<>())
      .def(
          "motion_cost",
          [](const CostFunction& c, const LocalPath* p) {
            return c.motionCost(*p);
          },
          py::arg("path"))
      .def("identity_cost", &CostFunction::identityCost)
      .def("combine", &CostFunction::combine, py::arg("a"), py::arg("b"));

  py::classh<PathLengthCost, CostFunction>(m, "PathLengthCost")
      .def(py::init<>());

  py::classh<CostmapCost, CostFunction>(m, "CostmapCost")
      .def(py::init<std::shared_ptr<const OccupancyGrid>, double, double>(),
           py::arg("costmap"), py::arg("weight"),
           py::arg("integration_step") = 0.05);

  // ---- Terrain ------------------------------------------------------------
  py::native_enum<TerrainType>(m, "TerrainType", "enum.IntEnum",
                               "Ground surfaces on construction / work sites.")
      .value("ASPHALT", TerrainType::kAsphalt)
      .value("CONCRETE_SLAB", TerrainType::kConcreteSlab)
      .value("STEEL_PLATE", TerrainType::kSteelPlate)
      .value("COMPACTED_GRAVEL", TerrainType::kCompactedGravel)
      .value("PACKED_DIRT", TerrainType::kPackedDirt)
      .value("GRASS", TerrainType::kGrass)
      .value("LOOSE_GRAVEL", TerrainType::kLooseGravel)
      .value("CRUSHED_ROCK", TerrainType::kCrushedRock)
      .value("SAND", TerrainType::kSand)
      .value("WET_CLAY", TerrainType::kWetClay)
      .value("RUBBLE", TerrainType::kRubble)
      .value("MUD", TerrainType::kMud)
      .value("STANDING_WATER", TerrainType::kStandingWater)
      .value("EXCAVATION", TerrainType::kExcavation)
      .value("STOCKPILE", TerrainType::kStockpile)
      .value("FRESH_CONCRETE", TerrainType::kFreshConcrete)
      .value("STRUCTURE", TerrainType::kStructure)
      .finalize();

  py::native_enum<SiteType>(m, "SiteType", "enum.Enum",
                            "Site layouts known to TerrainMap.random_site.")
      .value("ROAD_WORKS", SiteType::kRoadWorks)
      .value("BUILDING", SiteType::kBuilding)
      .value("DEMOLITION", SiteType::kDemolition)
      .value("QUARRY", SiteType::kQuarry)
      .value("PIPELINE", SiteType::kPipeline)
      .finalize();

  m.def("terrain_name", &terrainName, py::arg("terrain"));
  m.def("site_name", &siteName, py::arg("site"));
  m.def("default_terrain_properties", &defaultTerrainProperties,
        py::arg("terrain"));

  py::classh<TerrainProperties>(m, "TerrainProperties")
      .def(py::init([](double cost, bool traversable) {
             return TerrainProperties{cost, traversable};
           }),
           py::arg("cost") = 1.0, py::arg("traversable") = true)
      .def_readwrite("cost", &TerrainProperties::cost)
      .def_readwrite("traversable", &TerrainProperties::traversable)
      .def("__repr__", [](const TerrainProperties& p) {
        return "TerrainProperties(cost=" + std::to_string(p.cost) +
               ", traversable=" + (p.traversable ? "True" : "False") + ")";
      });

  py::classh<TerrainMap>(m, "TerrainMap")
      .def(py::init<std::size_t, std::size_t, double, double, double,
                    TerrainType>(),
           py::arg("width"), py::arg("height"), py::arg("resolution"),
           py::arg("origin_x") = 0.0, py::arg("origin_y") = 0.0,
           py::arg("fill") = TerrainType::kPackedDirt)
      .def_static(
          "from_array",
          [](py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> a,
             double resolution, double ox, double oy) {
            if (a.ndim() != 2) throw std::invalid_argument("expected 2D array");
            const auto h = static_cast<std::size_t>(a.shape(0));
            const auto w = static_cast<std::size_t>(a.shape(1));
            TerrainMap t(w, h, resolution, ox, oy);
            auto v = a.unchecked<2>();
            for (std::size_t r = 0; r < h; ++r) {
              for (std::size_t c = 0; c < w; ++c) {
                const auto x = v(r, c);
                if (x < 0 || x >= static_cast<std::int64_t>(kNumTerrainTypes))
                  throw std::invalid_argument("from_array: bad terrain value");
                t.set(c, r, static_cast<TerrainType>(x));
              }
            }
            return t;
          },
          py::arg("values"), py::arg("resolution"), py::arg("origin_x") = 0.0,
          py::arg("origin_y") = 0.0,
          "Build from an (height, width) array of TerrainType values; row 0 "
          "is the minimum y.")
      .def_static("random_site", &TerrainMap::randomSite, py::arg("site"),
                  py::arg("width"), py::arg("height"), py::arg("resolution"),
                  py::arg("seed"), py::arg("keep_clear") = std::vector<State>{},
                  py::arg("clear_radius") = 0.5, py::arg("origin_x") = 0.0,
                  py::arg("origin_y") = 0.0)
      .def("to_array",
           [](const TerrainMap& t) {
             py::array_t<std::uint8_t> out({t.height(), t.width()});
             auto m = out.mutable_unchecked<2>();
             for (std::size_t r = 0; r < t.height(); ++r)
               for (std::size_t c = 0; c < t.width(); ++c)
                 m(r, c) = static_cast<std::uint8_t>(t.at(c, r));
             return out;
           },
           "(height, width) uint8 array of TerrainType values.")
      .def("cost_array",
           [](const TerrainMap& t) {
             py::array_t<double> out({t.height(), t.width()});
             auto m = out.mutable_unchecked<2>();
             for (std::size_t r = 0; r < t.height(); ++r) {
               for (std::size_t c = 0; c < t.width(); ++c) {
                 const auto& p = t.properties(t.at(c, r));
                 m(r, c) = p.traversable ? p.cost
                                         : std::numeric_limits<double>::infinity();
               }
             }
             return out;
           },
           "(height, width) cost per unit length; inf where impassable.")
      .def("fill_rect", &TerrainMap::fillRect, py::arg("x_min"),
           py::arg("y_min"), py::arg("x_max"), py::arg("y_max"),
           py::arg("terrain"))
      .def("fill_disc", &TerrainMap::fillDisc, py::arg("cx"), py::arg("cy"),
           py::arg("radius"), py::arg("terrain"))
      .def("terrain_at", &TerrainMap::terrainAt, py::arg("x"), py::arg("y"))
      .def("cost_at", &TerrainMap::costAt, py::arg("x"), py::arg("y"),
           py::arg("outside") = std::numeric_limits<double>::infinity())
      .def("traversable_at", &TerrainMap::traversableAt, py::arg("x"),
           py::arg("y"))
      .def("properties", &TerrainMap::properties, py::arg("terrain"))
      .def("set_properties", &TerrainMap::setProperties, py::arg("terrain"),
           py::arg("properties"))
      .def("occupancy_grid", &TerrainMap::occupancyGrid)
      .def("normalized_cost_grid", &TerrainMap::normalizedCostGrid)
      .def_property_readonly("width", &TerrainMap::width)
      .def_property_readonly("height", &TerrainMap::height)
      .def_property_readonly("resolution", &TerrainMap::resolution)
      .def_property_readonly("origin", [](const TerrainMap& t) {
        return py::make_tuple(t.originX(), t.originY());
      });

  py::classh<TerrainCost, CostFunction>(m, "TerrainCost")
      .def(py::init<std::shared_ptr<const TerrainMap>, double, double>(),
           py::arg("terrain"), py::arg("integration_step") = 0.05,
           py::arg("blocked_cost") = 1e3);

  py::classh<TerrainCollisionChecker, CollisionChecker>(
      m, "TerrainCollisionChecker")
      .def(py::init<std::shared_ptr<const TerrainMap>, double>(),
           py::arg("terrain"), py::arg("resolution") = 0.05);

  // ---- Goal ---------------------------------------------------------------
  py::classh<Goal, PyGoal>(m, "Goal")
      .def(py::init<>())
      .def("is_satisfied", &Goal::isSatisfied, py::arg("state"))
      .def("sample", &Goal::sample, py::arg("rng"));

  py::classh<GoalState, Goal>(m, "GoalState")
      .def(py::init<std::shared_ptr<const StateSpace>, State, double>(),
           py::arg("space"), py::arg("state"), py::arg("tolerance"))
      .def_property_readonly("state", &GoalState::state)
      .def_property_readonly("tolerance", &GoalState::tolerance);

  // ---- Problem / results --------------------------------------------------
  py::classh<PlanningProblem>(m, "PlanningProblem")
      .def(py::init([](std::shared_ptr<const StateSpace> space,
                       std::shared_ptr<const CollisionChecker> checker,
                       std::shared_ptr<const CostFunction> cost, State start,
                       std::shared_ptr<const Goal> goal) {
             return PlanningProblem{std::move(space), std::move(checker),
                                    std::move(cost), std::move(start),
                                    std::move(goal)};
           }),
           py::arg("space"), py::arg("checker"), py::arg("cost"),
           py::arg("start"), py::arg("goal"))
      .def_readwrite("space", &PlanningProblem::space)
      .def_readwrite("checker", &PlanningProblem::checker)
      .def_readwrite("cost", &PlanningProblem::cost)
      .def_readwrite("start", &PlanningProblem::start)
      .def_readwrite("goal", &PlanningProblem::goal);

  py::classh<Termination>(m, "Termination")
      .def(py::init(&makeTermination), py::arg("max_iterations") = 0,
           py::arg("time_limit") = 0.0,
           py::arg("stop_on_first_solution") = false)
      .def_readwrite("max_iterations", &Termination::max_iterations)
      .def_readwrite("time_limit", &Termination::time_limit_s)
      .def_readwrite("stop_on_first_solution",
                     &Termination::stop_on_first_solution);

  py::native_enum<PlannerStatus>(m, "PlannerStatus", "enum.Enum")
      .value("SOLVED", PlannerStatus::kSolved)
      .value("NO_SOLUTION", PlannerStatus::kNoSolution)
      .value("INVALID_START", PlannerStatus::kInvalidStart)
      .value("INVALID_PROBLEM", PlannerStatus::kInvalidProblem)
      .finalize();

  py::classh<ConvergenceSample>(m, "ConvergenceSample")
      .def_readonly("time", &ConvergenceSample::time_s)
      .def_readonly("iteration", &ConvergenceSample::iteration)
      .def_readonly("tree_size", &ConvergenceSample::tree_size)
      .def_readonly("cost", &ConvergenceSample::cost);

  py::classh<PlannerResult>(m, "PlannerResult")
      .def(py::init<>())
      .def_readwrite("status", &PlannerResult::status)
      .def_property_readonly("solved", &PlannerResult::solved)
      .def_readwrite("cost", &PlannerResult::cost)
      .def_readwrite("iterations", &PlannerResult::iterations)
      .def_readwrite("tree_size", &PlannerResult::tree_size)
      .def_readwrite("elapsed", &PlannerResult::elapsed_s)
      .def_readwrite("first_solution_iteration",
                     &PlannerResult::first_solution_iteration)
      .def_readwrite("first_solution_time",
                     &PlannerResult::first_solution_time_s)
      .def_readwrite("segments", &PlannerResult::segments)
      .def_readwrite("convergence", &PlannerResult::convergence)
      .def_property_readonly(
          "waypoints",
          [](const PlannerResult& r) {
            return statesToArray(r.waypoints, stateDim(r.waypoints));
          },
          "Tree vertices from start to goal, as an N x dim array.")
      .def(
          "dense_path",
          [](const PlannerResult& r, double resolution) {
            const auto pts = r.densePath(resolution);
            return statesToArray(pts, stateDim(pts));
          },
          py::arg("resolution") = 0.05)
      .def(
          "convergence_array",
          [](const PlannerResult& r) {
            py::array_t<double> out({r.convergence.size(), std::size_t{4}});
            auto m = out.mutable_unchecked<2>();
            for (std::size_t i = 0; i < r.convergence.size(); ++i) {
              const auto& c = r.convergence[i];
              m(i, 0) = c.time_s;
              m(i, 1) = static_cast<double>(c.iteration);
              m(i, 2) = static_cast<double>(c.tree_size);
              m(i, 3) = c.cost;
            }
            return out;
          },
          "Convergence history as an N x 4 array of columns "
          "(time, iteration, tree_size, cost).")
      .def("__repr__", [](const PlannerResult& r) {
        return "<PlannerResult " + toString(r.status) +
               " cost=" + std::to_string(r.cost) +
               " iterations=" + std::to_string(r.iterations) +
               " tree_size=" + std::to_string(r.tree_size) + ">";
      });

  // ---- Planners -----------------------------------------------------------
  py::classh<Planner, PyPlanner>(m, "Planner")
      .def(py::init<>())
      .def("name", &Planner::name)
      .def("set_problem", &Planner::setProblem, py::arg("problem"))
      .def("solve", &Planner::solve, py::arg("termination"),
           py::call_guard<py::gil_scoped_release>())
      .def(
          "solve",
          [](Planner& p, std::size_t max_iterations, double time_limit,
             bool stop_on_first_solution) {
            const Termination t = makeTermination(
                max_iterations, time_limit, stop_on_first_solution);
            py::gil_scoped_release release;
            return p.solve(t);
          },
          py::kw_only(), py::arg("max_iterations") = 0,
          py::arg("time_limit") = 0.0,
          py::arg("stop_on_first_solution") = false)
      .def("clear", &Planner::clear);

  py::classh<RRTStarConfig>(m, "RRTStarConfig")
      .def(py::init([](double max_step, double goal_bias, double rewire_factor,
                       double max_rewire_radius, double min_rewire_radius,
                       std::uint64_t seed, std::size_t progress_interval) {
             RRTStarConfig c;
             c.max_step = max_step;
             c.goal_bias = goal_bias;
             c.rewire_factor = rewire_factor;
             c.max_rewire_radius = max_rewire_radius;
             c.min_rewire_radius = min_rewire_radius;
             c.seed = seed;
             c.progress_interval = progress_interval;
             return c;
           }),
           py::kw_only(), py::arg("max_step") = 0.5,
           py::arg("goal_bias") = 0.05, py::arg("rewire_factor") = 1.1,
           py::arg("max_rewire_radius") = 0.0,
           py::arg("min_rewire_radius") = 0.0, py::arg("seed") = 42,
           py::arg("progress_interval") = 0)
      .def_readwrite("max_step", &RRTStarConfig::max_step)
      .def_readwrite("goal_bias", &RRTStarConfig::goal_bias)
      .def_readwrite("rewire_factor", &RRTStarConfig::rewire_factor)
      .def_readwrite("max_rewire_radius", &RRTStarConfig::max_rewire_radius)
      .def_readwrite("min_rewire_radius", &RRTStarConfig::min_rewire_radius)
      .def_readwrite("seed", &RRTStarConfig::seed)
      .def_readwrite("progress_interval", &RRTStarConfig::progress_interval);

  py::classh<RRTStar, Planner>(m, "RRTStar")
      .def(py::init<RRTStarConfig>(), py::arg("config") = RRTStarConfig{})
      .def_property_readonly("config", &RRTStar::config)
      .def("rewire_radius", &RRTStar::rewireRadius, py::arg("n"))
      .def("tree_size", [](const RRTStar& p) { return p.tree().size(); })
      .def(
          "tree_states",
          [](const RRTStar& p) {
            const auto& t = p.tree();
            std::vector<State> s;
            s.reserve(t.size());
            for (const auto& n : t) s.push_back(n.state);
            return statesToArray(s, stateDim(s));
          },
          "Vertex states as an N x dim array.")
      .def(
          "tree_parents",
          [](const RRTStar& p) {
            const auto& t = p.tree();
            py::array_t<std::int64_t> out(t.size());
            auto m = out.mutable_unchecked<1>();
            for (std::size_t i = 0; i < t.size(); ++i) {
              m(i) = t[i].parent == RRTStar::kNoParent
                         ? -1
                         : static_cast<std::int64_t>(t[i].parent);
            }
            return out;
          },
          "Parent index of each vertex (-1 for the root).")
      .def(
          "tree_costs",
          [](const RRTStar& p) {
            const auto& t = p.tree();
            py::array_t<double> out(t.size());
            auto m = out.mutable_unchecked<1>();
            for (std::size_t i = 0; i < t.size(); ++i) m(i) = t[i].cost;
            return out;
          })
      .def(
          "tree_edges",
          [](const RRTStar& p, double resolution) {
            py::list edges;
            for (const auto& n : p.tree()) {
              if (!n.edge) continue;
              const double len = n.edge->length();
              const auto k = static_cast<std::size_t>(
                  std::max(1.0, std::ceil(len / resolution)));
              std::vector<State> pts;
              for (std::size_t i = 0; i <= k; ++i) {
                pts.push_back(n.edge->interpolate(
                    len * static_cast<double>(i) / static_cast<double>(k)));
              }
              edges.append(statesToArray(pts, stateDim(pts)));
            }
            return edges;
          },
          py::arg("resolution") = 0.1,
          "Every tree edge sampled as an array, for plotting.");

  // ---- Dubins -------------------------------------------------------------
  auto dm = m.def_submodule("dubins", "Closed-form Dubins shortest paths");
  py::native_enum<dubins::Word>(dm, "Word", "enum.Enum")
      .value("LSL", dubins::Word::kLSL)
      .value("LSR", dubins::Word::kLSR)
      .value("RSL", dubins::Word::kRSL)
      .value("RSR", dubins::Word::kRSR)
      .value("RLR", dubins::Word::kRLR)
      .value("LRL", dubins::Word::kLRL)
      .finalize();
  py::classh<dubins::Pose>(dm, "Pose")
      .def(py::init([](double x, double y, double theta) {
             return dubins::Pose{x, y, theta};
           }),
           py::arg("x") = 0.0, py::arg("y") = 0.0, py::arg("theta") = 0.0)
      .def_readwrite("x", &dubins::Pose::x)
      .def_readwrite("y", &dubins::Pose::y)
      .def_readwrite("theta", &dubins::Pose::theta)
      .def("__iter__",
           [](const dubins::Pose& p) {
             return py::iter(py::make_tuple(p.x, p.y, p.theta));
           })
      .def("__repr__", [](const dubins::Pose& p) {
        return "Pose(" + std::to_string(p.x) + ", " + std::to_string(p.y) +
               ", " + std::to_string(p.theta) + ")";
      });
  py::classh<dubins::Path>(dm, "Path")
      .def_readonly("start", &dubins::Path::start)
      .def_readonly("rho", &dubins::Path::rho)
      .def_readonly("word", &dubins::Path::word)
      .def_readonly("params", &dubins::Path::params)
      .def("length", &dubins::Path::length)
      .def("sample", &dubins::Path::sample, py::arg("s"))
      .def("end", &dubins::Path::end);
  dm.def("shortest_path", &dubins::shortestPath, py::arg("start"),
         py::arg("goal"), py::arg("rho"));
  dm.def("solve_word", &dubins::solveWord, py::arg("start"), py::arg("goal"),
         py::arg("rho"), py::arg("word"));
}
