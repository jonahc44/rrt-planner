#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "rrt/core/nearest_neighbors.hpp"
#include "rrt/core/planner.hpp"

namespace rrt {

struct RRTStarConfig {
  /// Maximum arc length of a single extension.
  double max_step = 0.5;
  /// Probability of sampling the goal instead of a uniform state.
  double goal_bias = 0.05;
  /// Multiplier on the theoretical minimum rewiring constant gamma*.
  /// Must be > 1 for asymptotic optimality.
  double rewire_factor = 1.1;
  /// Upper bound on the rewiring radius (0 = use max_step).
  double max_rewire_radius = 0.0;
  /// Lower bound on the rewiring radius, useful for spaces where short
  /// connections are rarely feasible (e.g. Dubins, set ~ turning radius).
  double min_rewire_radius = 0.0;
  std::uint64_t seed = 42;
};

/// RRT* (Karaman & Frazzoli, 2011) written purely against the StateSpace /
/// CollisionChecker / CostFunction interfaces. With a RealVector2DSpace it is
/// classic geometric RRT*; with a DubinsStateSpace every edge is an exact
/// Dubins curve, giving kinodynamic RRT* for a car with bounded curvature.
/// Handles asymmetric steering metrics.
class RRTStar final : public Planner {
 public:
  explicit RRTStar(RRTStarConfig config = {});

  std::string name() const override { return "RRT*"; }
  void setProblem(PlanningProblem problem) override;
  PlannerResult solve(const Termination& term) override;
  void clear() override;

  const RRTStarConfig& config() const { return config_; }

  struct Node {
    State state;
    std::size_t parent;  // kNoParent for the root
    LocalPathPtr edge;   // path parent -> this (null for the root)
    double edge_cost;    // motionCost(edge)
    double cost;         // cost-to-come
    std::vector<std::size_t> children;
  };
  static constexpr std::size_t kNoParent = static_cast<std::size_t>(-1);

  /// Read-only view of the tree (e.g. for visualisation).
  const std::vector<Node>& tree() const { return nodes_; }

  /// Current rewiring radius for a tree of n vertices.
  double rewireRadius(std::size_t n) const;

 private:
  void iterate();
  std::size_t addNode(State s, std::size_t parent, LocalPathPtr edge,
                      double edge_cost);
  void reparent(std::size_t node, std::size_t new_parent, LocalPathPtr edge,
                double edge_cost);
  void updateBestGoal();
  PlannerResult buildResult(PlannerStatus status) const;

  RRTStarConfig config_;
  PlanningProblem problem_;
  Rng rng_;
  std::vector<Node> nodes_;
  std::unique_ptr<NearestNeighbors> nn_;
  std::vector<std::size_t> goal_nodes_;
  std::size_t best_goal_ = kNoParent;
  double gamma_ = 0.0;
  std::size_t total_iterations_ = 0;
  std::size_t first_solution_iteration_ = 0;
  bool problem_valid_ = false;
};

}  // namespace rrt
