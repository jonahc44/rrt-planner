#include <algorithm>
#include <cmath>

#include "rrt/spaces/real_vector_2d_space.hpp"

namespace rrt {

RealVector2DSpace::RealVector2DSpace(double x_min, double x_max, double y_min,
                                     double y_max)
    : x_min_(x_min), x_max_(x_max), y_min_(y_min), y_max_(y_max) {}

State RealVector2DSpace::sampleUniform(Rng& rng) const {
  std::uniform_real_distribution<double> ux(x_min_, x_max_);
  std::uniform_real_distribution<double> uy(y_min_, y_max_);
  return {ux(rng), uy(rng)};
}

double RealVector2DSpace::distance(const State& from, const State& to) const {
  return std::hypot(to[0] - from[0], to[1] - from[1]);
}

LocalPathPtr RealVector2DSpace::connect(const State& from,
                                        const State& to) const {
  return std::make_shared<StraightLinePath>(from, to);
}

void RealVector2DSpace::embed(const State& s, double* out) const {
  out[0] = s[0];
  out[1] = s[1];
}

double RealVector2DSpace::measure() const {
  return (x_max_ - x_min_) * (y_max_ - y_min_);
}

StraightLinePath::StraightLinePath(State from, State to)
    : from_(std::move(from)), to_(std::move(to)) {
  double sq = 0.0;
  for (std::size_t i = 0; i < from_.size(); ++i) {
    const double d = to_[i] - from_[i];
    sq += d * d;
  }
  length_ = std::sqrt(sq);
}

State StraightLinePath::interpolate(double s) const {
  if (length_ <= 0.0) return from_;
  const double t = std::clamp(s / length_, 0.0, 1.0);
  State out(from_.size());
  for (std::size_t i = 0; i < from_.size(); ++i) {
    out[i] = from_[i] + t * (to_[i] - from_[i]);
  }
  return out;
}

}  // namespace rrt
