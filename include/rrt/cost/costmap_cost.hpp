#pragma once

#include <memory>

#include "rrt/collision/occupancy_grid.hpp"
#include "rrt/core/cost_function.hpp"

namespace rrt {

/// Integral of (1 + weight * c(x, y)) ds along the path, where c is the cost
/// map value in [0, 1]. weight = 0 reduces to path length; larger weights
/// trade length for staying in cheap regions.
class CostmapCost final : public CostFunction {
 public:
  CostmapCost(std::shared_ptr<const OccupancyGrid> costmap, double weight,
              double integration_step = 0.05)
      : map_(std::move(costmap)), weight_(weight), step_(integration_step) {}

  double motionCost(const LocalPath& path) const override;

 private:
  std::shared_ptr<const OccupancyGrid> map_;
  double weight_;
  double step_;
};

}  // namespace rrt
