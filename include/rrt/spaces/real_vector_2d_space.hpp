#pragma once

#include "rrt/core/state_space.hpp"

namespace rrt {

/// Holonomic point robot in a bounded rectangle of R^2. State = [x, y].
/// Steering is a straight line.
class RealVector2DSpace final : public StateSpace {
 public:
  RealVector2DSpace(double x_min, double x_max, double y_min, double y_max);

  std::size_t dimension() const override { return 2; }
  State sampleUniform(Rng& rng) const override;
  double distance(const State& from, const State& to) const override;
  LocalPathPtr connect(const State& from, const State& to) const override;

  std::size_t embeddingDimension() const override { return 2; }
  void embed(const State& s, double* out) const override;
  double measure() const override;

 private:
  double x_min_, x_max_, y_min_, y_max_;
};

/// Straight segment between two states of equal dimension.
class StraightLinePath final : public LocalPath {
 public:
  StraightLinePath(State from, State to);
  double length() const override { return length_; }
  State interpolate(double s) const override;

 private:
  State from_, to_;
  double length_;
};

}  // namespace rrt
