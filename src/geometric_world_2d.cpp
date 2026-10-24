#include "rrt/collision/geometric_world_2d.hpp"

#include <algorithm>

namespace rrt {

GeometricWorld2D::GeometricWorld2D(double x_min, double x_max, double y_min,
                                   double y_max, double robot_radius,
                                   double resolution)
    : CollisionChecker(resolution), x_min_(x_min), x_max_(x_max),
      y_min_(y_min), y_max_(y_max), robot_radius_(robot_radius) {}

bool GeometricWorld2D::isStateValid(const State& s) const {
  const double x = s[0], y = s[1], r = robot_radius_;
  if (x - r < x_min_ || x + r > x_max_ || y - r < y_min_ || y + r > y_max_) {
    return false;
  }
  for (const Circle& c : circles_) {
    const double dx = x - c.cx, dy = y - c.cy, rr = c.r + r;
    if (dx * dx + dy * dy <= rr * rr) return false;
  }
  for (const Box& b : boxes_) {
    // Distance from the disc centre to the closest point of the box.
    const double dx = x - std::clamp(x, b.x_min, b.x_max);
    const double dy = y - std::clamp(y, b.y_min, b.y_max);
    if (dx * dx + dy * dy <= r * r) return false;
  }
  return true;
}

}  // namespace rrt
