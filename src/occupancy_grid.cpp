#include "rrt/collision/occupancy_grid.hpp"

#include <cmath>
#include <stdexcept>

namespace rrt {

OccupancyGrid::OccupancyGrid(std::size_t width, std::size_t height,
                             double resolution, double origin_x,
                             double origin_y)
    : width_(width), height_(height), resolution_(resolution),
      origin_x_(origin_x), origin_y_(origin_y), cells_(width * height, 0.0) {}

OccupancyGrid OccupancyGrid::fromAscii(const std::vector<std::string>& rows,
                                       double resolution, double origin_x,
                                       double origin_y) {
  if (rows.empty()) throw std::invalid_argument("fromAscii: no rows");
  const std::size_t w = rows.front().size();
  const std::size_t h = rows.size();
  OccupancyGrid grid(w, h, resolution, origin_x, origin_y);
  for (std::size_t r = 0; r < h; ++r) {
    if (rows[r].size() != w) {
      throw std::invalid_argument("fromAscii: ragged rows");
    }
    const std::size_t row = h - 1 - r;  // first line is the top
    for (std::size_t col = 0; col < w; ++col) {
      const char c = rows[r][col];
      double v = 0.0;
      if (c == '#') {
        v = 1.0;
      } else if (c >= '0' && c <= '9') {
        v = (c - '0') / 9.0;
      } else if (c != '.' && c != ' ') {
        throw std::invalid_argument("fromAscii: unknown cell character");
      }
      grid.at(col, row) = v;
    }
  }
  return grid;
}

bool OccupancyGrid::inBounds(double x, double y) const {
  return x >= origin_x_ && y >= origin_y_ &&
         x < origin_x_ + static_cast<double>(width_) * resolution_ &&
         y < origin_y_ + static_cast<double>(height_) * resolution_;
}

double OccupancyGrid::valueAt(double x, double y, double outside) const {
  if (!inBounds(x, y)) return outside;
  const auto col = static_cast<std::size_t>((x - origin_x_) / resolution_);
  const auto row = static_cast<std::size_t>((y - origin_y_) / resolution_);
  if (col >= width_ || row >= height_) return outside;
  return at(col, row);
}

}  // namespace rrt
