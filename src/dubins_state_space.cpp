#include "rrt/spaces/dubins_state_space.hpp"

namespace rrt {
namespace {

constexpr double kPi = 3.14159265358979323846;

dubins::Pose toPose(const State& s) { return {s[0], s[1], s[2]}; }

}  // namespace

DubinsStateSpace::DubinsStateSpace(double turning_radius, double x_min,
                                   double x_max, double y_min, double y_max)
    : rho_(turning_radius), x_min_(x_min), x_max_(x_max), y_min_(y_min),
      y_max_(y_max) {}

State DubinsStateSpace::sampleUniform(Rng& rng) const {
  std::uniform_real_distribution<double> ux(x_min_, x_max_);
  std::uniform_real_distribution<double> uy(y_min_, y_max_);
  std::uniform_real_distribution<double> ut(-kPi, kPi);
  return {ux(rng), uy(rng), ut(rng)};
}

double DubinsStateSpace::distance(const State& from, const State& to) const {
  return dubins::shortestPath(toPose(from), toPose(to), rho_).length();
}

LocalPathPtr DubinsStateSpace::connect(const State& from,
                                       const State& to) const {
  return std::make_shared<DubinsLocalPath>(
      dubins::shortestPath(toPose(from), toPose(to), rho_));
}

void DubinsStateSpace::embed(const State& s, double* out) const {
  out[0] = s[0];
  out[1] = s[1];
}

double DubinsStateSpace::measure() const {
  // Heading measured in arc-length units (rho * angle) so the volume is
  // commensurate with the distance metric.
  return (x_max_ - x_min_) * (y_max_ - y_min_) * 2.0 * kPi * rho_;
}

State DubinsLocalPath::interpolate(double s) const {
  const dubins::Pose p = path_.sample(s);
  return {p.x, p.y, p.theta};
}

}  // namespace rrt
