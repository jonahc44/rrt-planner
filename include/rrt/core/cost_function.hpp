#pragma once

#include "rrt/core/local_path.hpp"

namespace rrt {

/// Objective that the optimizing planner minimises. Costs must be
/// non-negative and additive along a path. Subclass to plug in a new
/// objective (path length, cost maps, energy, time...).
class CostFunction {
 public:
  virtual ~CostFunction() = default;

  /// Cost of traversing a single local path.
  virtual double motionCost(const LocalPath& path) const = 0;

  /// Cost of the start state (the root of the tree).
  virtual double identityCost() const { return 0.0; }

  /// Cost of the concatenation of two path segments.
  virtual double combine(double a, double b) const { return a + b; }
};

/// Minimise arc length.
class PathLengthCost final : public CostFunction {
 public:
  double motionCost(const LocalPath& path) const override {
    return path.length();
  }
};

}  // namespace rrt
