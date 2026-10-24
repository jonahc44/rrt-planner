#include <algorithm>
#include <cmath>

#include "rrt/core/collision_checker.hpp"

namespace rrt {

bool CollisionChecker::isMotionValid(const LocalPath& path) const {
  const double len = path.length();
  const auto n = static_cast<std::size_t>(
      std::max(1.0, std::ceil(len / resolution_)));
  // Endpoints first: cheap early-out for the common case of an invalid tip.
  if (!isStateValid(path.interpolate(len))) return false;
  for (std::size_t i = 0; i < n; ++i) {
    if (!isStateValid(path.interpolate(len * static_cast<double>(i) /
                                       static_cast<double>(n)))) {
      return false;
    }
  }
  return true;
}

}  // namespace rrt
