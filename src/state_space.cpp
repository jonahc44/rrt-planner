#include <algorithm>
#include <cmath>

#include "rrt/core/planner.hpp"
#include "rrt/core/state_space.hpp"

namespace rrt {

TruncatedPath::TruncatedPath(LocalPathPtr inner, double max_length)
    : inner_(std::move(inner)),
      length_(std::min(inner_->length(), std::max(0.0, max_length))) {}

State TruncatedPath::interpolate(double s) const {
  return inner_->interpolate(std::clamp(s, 0.0, length_));
}

LocalPathPtr StateSpace::steer(const State& from, const State& toward,
                               double max_length) const {
  LocalPathPtr full = connect(from, toward);
  if (!full || full->length() <= max_length) return full;
  return std::make_shared<TruncatedPath>(std::move(full), max_length);
}

std::string toString(PlannerStatus status) {
  switch (status) {
    case PlannerStatus::kSolved: return "solved";
    case PlannerStatus::kNoSolution: return "no solution";
    case PlannerStatus::kInvalidStart: return "invalid start";
    case PlannerStatus::kInvalidProblem: return "invalid problem";
  }
  return "unknown";
}

std::vector<State> PlannerResult::densePath(double resolution) const {
  std::vector<State> out;
  if (waypoints.empty()) return out;
  out.push_back(waypoints.front());
  for (const auto& seg : segments) {
    const double len = seg->length();
    const auto n = static_cast<std::size_t>(
        std::max(1.0, std::ceil(len / resolution)));
    for (std::size_t i = 1; i <= n; ++i) {
      out.push_back(seg->interpolate(len * static_cast<double>(i) /
                                     static_cast<double>(n)));
    }
  }
  return out;
}

}  // namespace rrt
