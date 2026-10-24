#pragma once

#include "rrt/core/state_space.hpp"
#include "rrt/spaces/dubins.hpp"

namespace rrt {

/// Forward-only car with a minimum turning radius. State = [x, y, theta],
/// theta in [-pi, pi). Steering follows the shortest Dubins curve, so every
/// edge of the tree is kinematically feasible.
class DubinsStateSpace final : public StateSpace {
 public:
  DubinsStateSpace(double turning_radius, double x_min, double x_max,
                   double y_min, double y_max);

  std::size_t dimension() const override { return 3; }
  State sampleUniform(Rng& rng) const override;
  double distance(const State& from, const State& to) const override;
  LocalPathPtr connect(const State& from, const State& to) const override;

  /// (x, y): a Dubins path is never shorter than the straight line between
  /// its endpoints, so this is a valid lower-bound embedding.
  std::size_t embeddingDimension() const override { return 2; }
  void embed(const State& s, double* out) const override;
  double measure() const override;

  double turningRadius() const { return rho_; }

 private:
  double rho_;
  double x_min_, x_max_, y_min_, y_max_;
};

class DubinsLocalPath final : public LocalPath {
 public:
  explicit DubinsLocalPath(dubins::Path path) : path_(path) {}
  double length() const override { return path_.length(); }
  State interpolate(double s) const override;
  const dubins::Path& dubinsPath() const { return path_; }

 private:
  dubins::Path path_;
};

}  // namespace rrt
