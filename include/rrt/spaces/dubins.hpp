#pragma once

#include <array>
#include <optional>

namespace rrt::dubins {

/// SE(2) pose.
struct Pose {
  double x = 0.0, y = 0.0, theta = 0.0;
};

enum class Segment { kLeft, kStraight, kRight };

enum class Word { kLSL, kLSR, kRSL, kRSR, kRLR, kLRL };

/// A shortest path for a forward-only car with minimum turning radius `rho`
/// (Dubins, 1957). It is one of six "words" of three segments.
struct Path {
  Pose start;
  double rho = 1.0;
  Word word = Word::kLSL;
  /// Segment lengths normalised by rho (i.e. turn angles for L/R segments).
  std::array<double, 3> params{};

  double length() const { return (params[0] + params[1] + params[2]) * rho; }

  /// Pose after travelling arc length s from the start (clamped).
  Pose sample(double s) const;
  Pose end() const { return sample(length()); }
};

std::array<Segment, 3> segments(Word w);
const char* toString(Word w);

/// Path for a single word, if that word is feasible for the given endpoints.
std::optional<Path> solveWord(const Pose& from, const Pose& to, double rho,
                              Word w);

/// Shortest Dubins path from `from` to `to`. Always exists for rho > 0.
Path shortestPath(const Pose& from, const Pose& to, double rho);

/// Wrap to [0, 2*pi).
double mod2pi(double a);
/// Wrap to [-pi, pi).
double wrapToPi(double a);

}  // namespace rrt::dubins
