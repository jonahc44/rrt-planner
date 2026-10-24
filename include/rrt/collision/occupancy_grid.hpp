#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "rrt/core/collision_checker.hpp"

namespace rrt {

/// Row-major 2D grid of cell values in [0, 1]. Used both as an occupancy map
/// (for collision checking) and as a cost map (for the objective).
/// Cell (col, row) covers [origin_x + col*res, origin_x + (col+1)*res) x ...
class OccupancyGrid {
 public:
  OccupancyGrid(std::size_t width, std::size_t height, double resolution,
                double origin_x = 0.0, double origin_y = 0.0);

  /// Build from ASCII art: '#' = 1.0, '.' or ' ' = 0.0, digits '0'-'9' =
  /// value/9. The first line is the top row (max y).
  static OccupancyGrid fromAscii(const std::vector<std::string>& rows,
                                 double resolution, double origin_x = 0.0,
                                 double origin_y = 0.0);

  std::size_t width() const { return width_; }
  std::size_t height() const { return height_; }
  double resolution() const { return resolution_; }
  double originX() const { return origin_x_; }
  double originY() const { return origin_y_; }

  double& at(std::size_t col, std::size_t row) {
    return cells_[row * width_ + col];
  }
  double at(std::size_t col, std::size_t row) const {
    return cells_[row * width_ + col];
  }

  bool inBounds(double x, double y) const;
  /// Value of the cell containing (x, y); `outside` if out of bounds.
  double valueAt(double x, double y, double outside = 1.0) const;

 private:
  std::size_t width_, height_;
  double resolution_, origin_x_, origin_y_;
  std::vector<double> cells_;
};

/// Point robot on an occupancy grid: a state is valid if its cell value is
/// below `occupied_threshold`.
class GridCollisionChecker final : public CollisionChecker {
 public:
  GridCollisionChecker(std::shared_ptr<const OccupancyGrid> grid,
                       double occupied_threshold = 0.5,
                       double resolution = 0.05)
      : CollisionChecker(resolution), grid_(std::move(grid)),
        threshold_(occupied_threshold) {}

  bool isStateValid(const State& s) const override {
    return grid_->valueAt(s[0], s[1]) < threshold_;
  }

 private:
  std::shared_ptr<const OccupancyGrid> grid_;
  double threshold_;
};

}  // namespace rrt
