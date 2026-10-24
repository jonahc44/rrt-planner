#include "rrt/planners/rrt_star.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <tuple>

namespace rrt {
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kMinEdgeLength = 1e-9;
constexpr double kCostEpsilon = 1e-9;

/// Volume of the unit ball in R^d.
double unitBallVolume(std::size_t d) {
  const double dd = static_cast<double>(d);
  return std::pow(kPi, dd / 2.0) / std::tgamma(dd / 2.0 + 1.0);
}

}  // namespace

RRTStar::RRTStar(RRTStarConfig config)
    : config_(config), rng_(config.seed) {}

void RRTStar::setProblem(PlanningProblem problem) {
  problem_ = std::move(problem);
  problem_valid_ = problem_.space && problem_.checker && problem_.cost &&
                   problem_.goal &&
                   problem_.start.size() == problem_.space->dimension();
  if (problem_valid_) {
    // gamma* from Karaman & Frazzoli (2011), Theorem 38.
    const auto d = static_cast<double>(problem_.space->dimension());
    gamma_ = config_.rewire_factor * 2.0 * std::pow(1.0 + 1.0 / d, 1.0 / d) *
             std::pow(problem_.space->measure() /
                          unitBallVolume(problem_.space->dimension()),
                      1.0 / d);
  }
  clear();
}

void RRTStar::clear() {
  nodes_.clear();
  goal_nodes_.clear();
  best_goal_ = kNoParent;
  total_iterations_ = 0;
  first_solution_iteration_ = 0;
  rng_.seed(config_.seed);
  nn_.reset();
  if (!problem_valid_) return;
  nn_ = std::make_unique<KdTreeNearestNeighbors>(problem_.space);
  if (problem_.checker->isStateValid(problem_.start)) {
    nodes_.push_back(Node{problem_.start, kNoParent, nullptr, 0.0,
                          problem_.cost->identityCost(), {}});
    nn_->add(problem_.start);
    if (problem_.goal->isSatisfied(problem_.start)) {
      goal_nodes_.push_back(0);
      best_goal_ = 0;
    }
  }
}

double RRTStar::rewireRadius(std::size_t n) const {
  const double upper =
      config_.max_rewire_radius > 0.0 ? config_.max_rewire_radius
                                      : config_.max_step;
  double r = upper;
  if (n > 1) {
    const auto nn = static_cast<double>(n);
    const auto d = static_cast<double>(problem_.space->dimension());
    r = std::min(upper, gamma_ * std::pow(std::log(nn) / nn, 1.0 / d));
  }
  return std::max(r, config_.min_rewire_radius);
}

PlannerResult RRTStar::solve(const Termination& term) {
  if (!problem_valid_ ||
      (term.max_iterations == 0 && term.time_limit_s <= 0.0)) {
    return buildResult(PlannerStatus::kInvalidProblem);
  }
  if (nodes_.empty()) return buildResult(PlannerStatus::kInvalidStart);

  using Clock = std::chrono::steady_clock;
  const auto t0 = Clock::now();
  const auto elapsed = [&] {
    return std::chrono::duration<double>(Clock::now() - t0).count();
  };

  std::size_t iters = 0;
  while (true) {
    if (term.max_iterations > 0 && iters >= term.max_iterations) break;
    if (term.time_limit_s > 0.0 && elapsed() >= term.time_limit_s) break;
    if (term.stop_on_first_solution && best_goal_ != kNoParent) break;
    iterate();
    ++iters;
    ++total_iterations_;
    if (best_goal_ != kNoParent && first_solution_iteration_ == 0) {
      first_solution_iteration_ = total_iterations_;
    }
  }

  PlannerResult result = buildResult(best_goal_ != kNoParent
                                         ? PlannerStatus::kSolved
                                         : PlannerStatus::kNoSolution);
  result.iterations = iters;
  result.elapsed_s = elapsed();
  return result;
}

void RRTStar::iterate() {
  const StateSpace& space = *problem_.space;
  const CollisionChecker& checker = *problem_.checker;
  const CostFunction& cost = *problem_.cost;

  // 1. Sample.
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  const State x_rand = unit(rng_) < config_.goal_bias
                           ? problem_.goal->sample(rng_)
                           : space.sampleUniform(rng_);

  // 2. Extend from the nearest vertex.
  const std::size_t x_nearest = nn_->nearest(x_rand);
  LocalPathPtr steer_path =
      space.steer(nodes_[x_nearest].state, x_rand, config_.max_step);
  if (!steer_path || steer_path->length() < kMinEdgeLength) return;
  if (!checker.isMotionValid(*steer_path)) return;
  State x_new = steer_path->end();

  // 3. Choose the cheapest collision-free parent among vertices that can
  //    reach x_new within the rewiring radius.
  const double r = rewireRadius(nodes_.size() + 1);
  std::size_t best_parent = x_nearest;
  LocalPathPtr best_edge = steer_path;
  double best_edge_cost = cost.motionCost(*steer_path);
  double best_cost = cost.combine(nodes_[x_nearest].cost, best_edge_cost);

  struct Candidate {
    double total;
    double edge_cost;
    std::size_t id;
    LocalPathPtr edge;
  };
  std::vector<Candidate> candidates;
  for (std::size_t id : nn_->nearTo(x_new, r)) {
    if (id == x_nearest) continue;
    LocalPathPtr edge = space.connect(nodes_[id].state, x_new);
    if (!edge) continue;
    const double ec = cost.motionCost(*edge);
    const double total = cost.combine(nodes_[id].cost, ec);
    if (total < best_cost) candidates.push_back({total, ec, id, std::move(edge)});
  }
  // Collision checking is the expensive part: check lazily, best first.
  std::sort(candidates.begin(), candidates.end(),
            [](const Candidate& a, const Candidate& b) {
              return a.total < b.total;
            });
  for (Candidate& c : candidates) {
    if (checker.isMotionValid(*c.edge)) {
      best_parent = c.id;
      best_edge = std::move(c.edge);
      best_edge_cost = c.edge_cost;
      best_cost = c.total;
      break;
    }
  }

  const std::size_t new_id =
      addNode(std::move(x_new), best_parent, std::move(best_edge),
              best_edge_cost);
  const State& new_state = nodes_[new_id].state;

  // 4. Rewire: route neighbours through x_new where that is cheaper.
  for (std::size_t id : nn_->nearFrom(new_state, r)) {
    if (id == new_id || id == best_parent) continue;
    LocalPathPtr edge = space.connect(new_state, nodes_[id].state);
    if (!edge) continue;
    const double ec = cost.motionCost(*edge);
    const double total = cost.combine(nodes_[new_id].cost, ec);
    if (total + kCostEpsilon < nodes_[id].cost &&
        checker.isMotionValid(*edge)) {
      reparent(id, new_id, std::move(edge), ec);
    }
  }

  if (problem_.goal->isSatisfied(new_state)) goal_nodes_.push_back(new_id);
  updateBestGoal();
}

std::size_t RRTStar::addNode(State s, std::size_t parent, LocalPathPtr edge,
                             double edge_cost) {
  const std::size_t id = nodes_.size();
  const double c = problem_.cost->combine(nodes_[parent].cost, edge_cost);
  nn_->add(s);
  nodes_.push_back(Node{std::move(s), parent, std::move(edge), edge_cost, c, {}});
  nodes_[parent].children.push_back(id);
  return id;
}

void RRTStar::reparent(std::size_t node, std::size_t new_parent,
                       LocalPathPtr edge, double edge_cost) {
  Node& n = nodes_[node];
  auto& siblings = nodes_[n.parent].children;
  siblings.erase(std::find(siblings.begin(), siblings.end(), node));
  n.parent = new_parent;
  n.edge = std::move(edge);
  n.edge_cost = edge_cost;
  nodes_[new_parent].children.push_back(node);

  // Propagate the new cost-to-come through the subtree.
  std::vector<std::size_t> stack{node};
  while (!stack.empty()) {
    const std::size_t id = stack.back();
    stack.pop_back();
    Node& cur = nodes_[id];
    cur.cost = problem_.cost->combine(nodes_[cur.parent].cost, cur.edge_cost);
    stack.insert(stack.end(), cur.children.begin(), cur.children.end());
  }
}

void RRTStar::updateBestGoal() {
  double best = std::numeric_limits<double>::infinity();
  for (std::size_t id : goal_nodes_) {
    if (nodes_[id].cost < best) {
      best = nodes_[id].cost;
      best_goal_ = id;
    }
  }
}

PlannerResult RRTStar::buildResult(PlannerStatus status) const {
  PlannerResult result;
  result.status = status;
  result.tree_size = nodes_.size();
  result.first_solution_iteration = first_solution_iteration_;
  if (status != PlannerStatus::kSolved) return result;

  for (std::size_t id = best_goal_; id != kNoParent; id = nodes_[id].parent) {
    result.waypoints.push_back(nodes_[id].state);
    if (nodes_[id].edge) result.segments.push_back(nodes_[id].edge);
  }
  std::reverse(result.waypoints.begin(), result.waypoints.end());
  std::reverse(result.segments.begin(), result.segments.end());
  result.cost = nodes_[best_goal_].cost;
  return result;
}

}  // namespace rrt
