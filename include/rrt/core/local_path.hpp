#pragma once

#include <memory>

#include "rrt/core/types.hpp"

namespace rrt {

/// A feasible motion between two states, produced by a StateSpace's steering
/// function. Parameterised by arc length s in [0, length()].
class LocalPath {
 public:
  virtual ~LocalPath() = default;

  virtual double length() const = 0;

  /// State at arc length s; s is clamped to [0, length()].
  virtual State interpolate(double s) const = 0;

  State start() const { return interpolate(0.0); }
  State end() const { return interpolate(length()); }
};

using LocalPathPtr = std::shared_ptr<const LocalPath>;

/// Wraps another path and exposes only its first `max_length` units. Used by
/// the default StateSpace::steer() so any space gets truncated steering for
/// free.
class TruncatedPath final : public LocalPath {
 public:
  TruncatedPath(LocalPathPtr inner, double max_length);

  double length() const override { return length_; }
  State interpolate(double s) const override;

 private:
  LocalPathPtr inner_;
  double length_;
};

}  // namespace rrt
