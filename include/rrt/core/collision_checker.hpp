#pragma once

#include "rrt/core/local_path.hpp"
#include "rrt/core/types.hpp"

namespace rrt {

/// Answers "is this state / motion free?". Subclass to plug in a new
/// environment representation (geometric primitives, occupancy grids, a
/// signed distance field, an FCL scene...).
class CollisionChecker {
 public:
  /// `resolution` is the arc-length step used by the default motion check.
  explicit CollisionChecker(double resolution = 0.05)
      : resolution_(resolution) {}
  virtual ~CollisionChecker() = default;

  virtual bool isStateValid(const State& s) const = 0;

  /// Default: sample the path every `resolution()` units of arc length and
  /// check each state. Override if you can do better (e.g. swept volumes).
  virtual bool isMotionValid(const LocalPath& path) const;

  double resolution() const { return resolution_; }
  void setResolution(double r) { resolution_ = r; }

 private:
  double resolution_;
};

}  // namespace rrt
