#pragma once

#include <memory>

#include "rrt/collision/geometric_world_2d.hpp"

namespace rrt::examples {

/// 10 x 10 world with a wall that has a gap, plus a few round obstacles.
inline std::shared_ptr<GeometricWorld2D> makeDemoWorld(double robot_radius) {
  auto world = std::make_shared<GeometricWorld2D>(0.0, 10.0, 0.0, 10.0,
                                                  robot_radius, 0.02);
  world->addBox({4.5, 0.0, 5.5, 6.0});
  world->addBox({4.5, 7.8, 5.5, 10.0});
  world->addCircle({2.2, 6.5, 1.0});
  world->addCircle({7.5, 3.0, 1.2});
  world->addCircle({2.5, 2.5, 0.8});
  world->addCircle({8.0, 7.5, 0.7});
  return world;
}

}  // namespace rrt::examples
