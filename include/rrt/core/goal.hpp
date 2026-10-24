#pragma once

#include <memory>

#include "rrt/core/state_space.hpp"
#include "rrt/core/types.hpp"

namespace rrt {

/// A goal region. Planners bias sampling toward sample() and accept any state
/// for which isSatisfied() holds.
class Goal {
 public:
  virtual ~Goal() = default;
  virtual bool isSatisfied(const State& s) const = 0;
  virtual State sample(Rng& rng) const = 0;
};

/// Reach a specific state to within `tolerance` (measured by the space's
/// distance function, from s to the goal).
class GoalState final : public Goal {
 public:
  GoalState(std::shared_ptr<const StateSpace> space, State goal,
            double tolerance)
      : space_(std::move(space)), goal_(std::move(goal)),
        tolerance_(tolerance) {}

  bool isSatisfied(const State& s) const override {
    return space_->distance(s, goal_) <= tolerance_;
  }
  State sample(Rng&) const override { return goal_; }

  const State& state() const { return goal_; }
  double tolerance() const { return tolerance_; }

 private:
  std::shared_ptr<const StateSpace> space_;
  State goal_;
  double tolerance_;
};

}  // namespace rrt
