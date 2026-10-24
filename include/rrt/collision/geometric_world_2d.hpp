#pragma once

#include <vector>

#include "rrt/core/collision_checker.hpp"

namespace rrt {

struct Circle {
  double cx, cy, r;
};

/// Axis-aligned box.
struct Box {
  double x_min, y_min, x_max, y_max;
};

/// Planar world made of circles and boxes inside rectangular bounds. The
/// robot is a disc of radius `robot_radius` centred at (state[0], state[1]).
class GeometricWorld2D final : public CollisionChecker {
 public:
  GeometricWorld2D(double x_min, double x_max, double y_min, double y_max,
                   double robot_radius = 0.0, double resolution = 0.05);

  void addCircle(Circle c) { circles_.push_back(c); }
  void addBox(Box b) { boxes_.push_back(b); }

  bool isStateValid(const State& s) const override;

  const std::vector<Circle>& circles() const { return circles_; }
  const std::vector<Box>& boxes() const { return boxes_; }
  double xMin() const { return x_min_; }
  double xMax() const { return x_max_; }
  double yMin() const { return y_min_; }
  double yMax() const { return y_max_; }

 private:
  double x_min_, x_max_, y_min_, y_max_;
  double robot_radius_;
  std::vector<Circle> circles_;
  std::vector<Box> boxes_;
};

}  // namespace rrt
