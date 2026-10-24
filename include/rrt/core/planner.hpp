#pragma once

#include <cstddef>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "rrt/core/collision_checker.hpp"
#include "rrt/core/cost_function.hpp"
#include "rrt/core/goal.hpp"
#include "rrt/core/state_space.hpp"

namespace rrt {

/// Everything a planner needs to know about the problem. Swap any of the
/// components to change vehicle, environment, or objective.
struct PlanningProblem {
  std::shared_ptr<const StateSpace> space;
  std::shared_ptr<const CollisionChecker> checker;
  std::shared_ptr<const CostFunction> cost;
  State start;
  std::shared_ptr<const Goal> goal;
};

/// When to stop planning. A limit of 0 means "no limit" for that criterion;
/// at least one must be set.
struct Termination {
  std::size_t max_iterations = 0;
  double time_limit_s = 0.0;
  bool stop_on_first_solution = false;
};

/// One point on the anytime convergence curve. Times and iteration counts
/// are cumulative since the tree was last cleared.
struct ConvergenceSample {
  double time_s = 0.0;
  std::size_t iteration = 0;
  std::size_t tree_size = 0;
  /// Best solution cost so far (infinity before the first solution).
  double cost = std::numeric_limits<double>::infinity();
};

enum class PlannerStatus { kSolved, kNoSolution, kInvalidStart, kInvalidProblem };

std::string toString(PlannerStatus status);

struct PlannerResult {
  PlannerStatus status = PlannerStatus::kNoSolution;
  /// Tree vertices from start to goal.
  std::vector<State> waypoints;
  /// The local paths joining consecutive waypoints (waypoints.size() - 1).
  std::vector<LocalPathPtr> segments;
  double cost = std::numeric_limits<double>::infinity();
  std::size_t iterations = 0;
  std::size_t tree_size = 0;
  double elapsed_s = 0.0;
  /// Iteration at which the first solution was found (0 if none).
  std::size_t first_solution_iteration = 0;
  /// Cumulative planning time at which the first solution was found
  /// (infinity if none).
  double first_solution_time_s = std::numeric_limits<double>::infinity();
  /// Cost history since the tree was last cleared. Holds a sample at every
  /// improvement, plus periodic samples if the planner is configured to
  /// record them.
  std::vector<ConvergenceSample> convergence;

  bool solved() const { return status == PlannerStatus::kSolved; }

  /// Densely sampled path, spaced at most `resolution` apart.
  std::vector<State> densePath(double resolution) const;
};

class Planner {
 public:
  virtual ~Planner() = default;

  virtual std::string name() const = 0;

  /// Set (or replace) the problem. Clears any previous planning state.
  virtual void setProblem(PlanningProblem problem) = 0;

  /// Run until `term` is reached. Calling solve() again continues to refine
  /// the same tree.
  virtual PlannerResult solve(const Termination& term) = 0;

  /// Discard the search tree but keep the problem.
  virtual void clear() = 0;
};

}  // namespace rrt
