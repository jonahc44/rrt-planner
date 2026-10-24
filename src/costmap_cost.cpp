#include "rrt/cost/costmap_cost.hpp"

#include <algorithm>
#include <cmath>

namespace rrt {

double CostmapCost::motionCost(const LocalPath& path) const {
  const double len = path.length();
  if (len <= 0.0) return 0.0;
  const auto n = static_cast<std::size_t>(std::max(1.0, std::ceil(len / step_)));
  const double ds = len / static_cast<double>(n);
  // Midpoint rule over n sub-intervals.
  double total = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const State s = path.interpolate((static_cast<double>(i) + 0.5) * ds);
    total += 1.0 + weight_ * map_->valueAt(s[0], s[1], 1.0);
  }
  return total * ds;
}

}  // namespace rrt
