#pragma once

// Minimal SVG output for the examples: world, search tree, and solution.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>

#include "rrt/collision/geometric_world_2d.hpp"
#include "rrt/planners/rrt_star.hpp"

namespace rrt::examples {

class SvgWriter {
 public:
  SvgWriter(const GeometricWorld2D& world, double pixels_per_unit = 50.0)
      : world_(world), scale_(pixels_per_unit) {}

  void drawWorld() {
    for (const Circle& c : world_.circles()) {
      body_ += fmt("<circle cx='%.2f' cy='%.2f' r='%.2f' fill='#555'/>\n",
                   px(c.cx), py(c.cy), c.r * scale_);
    }
    for (const Box& b : world_.boxes()) {
      body_ += fmt(
          "<rect x='%.2f' y='%.2f' width='%.2f' height='%.2f' fill='#555'/>\n",
          px(b.x_min), py(b.y_max), (b.x_max - b.x_min) * scale_,
          (b.y_max - b.y_min) * scale_);
    }
  }

  void drawTree(const RRTStar& planner, double resolution) {
    for (const auto& node : planner.tree()) {
      if (node.edge) drawPath(*node.edge, resolution, "#9ab", 0.6);
    }
  }

  void drawPath(const LocalPath& path, double resolution,
                const char* color, double width) {
    std::string pts;
    const double len = path.length();
    const int n = std::max(1, static_cast<int>(len / resolution));
    for (int i = 0; i <= n; ++i) {
      const State s = path.interpolate(len * i / n);
      pts += fmt("%.2f,%.2f ", px(s[0]), py(s[1]));
    }
    body_ += fmt("<polyline points='%s' fill='none' stroke='%s' "
                 "stroke-width='%.2f'/>\n",
                 pts.c_str(), color, width);
  }

  void drawSolution(const PlannerResult& result, double resolution) {
    for (const auto& seg : result.segments) {
      drawPath(*seg, resolution, "#d33", 3.0);
    }
  }

  /// A marker at a state; draws a heading tick if the state has a third
  /// component.
  void drawPose(const State& s, const char* color) {
    body_ += fmt("<circle cx='%.2f' cy='%.2f' r='6' fill='%s'/>\n", px(s[0]),
                 py(s[1]), color);
    if (s.size() >= 3) {
      body_ += fmt("<line x1='%.2f' y1='%.2f' x2='%.2f' y2='%.2f' "
                   "stroke='%s' stroke-width='3'/>\n",
                   px(s[0]), py(s[1]), px(s[0] + 0.6 * std::cos(s[2])),
                   py(s[1] + 0.6 * std::sin(s[2])), color);
    }
  }

  bool save(const std::string& filename) const {
    std::ofstream out(filename);
    if (!out) return false;
    const double w = (world_.xMax() - world_.xMin()) * scale_;
    const double h = (world_.yMax() - world_.yMin()) * scale_;
    out << fmt("<svg xmlns='http://www.w3.org/2000/svg' width='%.0f' "
               "height='%.0f'>\n<rect width='100%%' height='100%%' "
               "fill='white' stroke='black'/>\n",
               w, h)
        << body_ << "</svg>\n";
    return static_cast<bool>(out);
  }

 private:
  double px(double x) const { return (x - world_.xMin()) * scale_; }
  double py(double y) const { return (world_.yMax() - y) * scale_; }

  template <typename... Args>
  static std::string fmt(const char* f, Args... args) {
    const int n = std::snprintf(nullptr, 0, f, args...);
    std::string s(static_cast<std::size_t>(n) + 1, '\0');
    std::snprintf(s.data(), s.size(), f, args...);
    s.pop_back();
    return s;
  }

  const GeometricWorld2D& world_;
  double scale_;
  std::string body_;
};

}  // namespace rrt::examples
