// Dubins shortest paths, following the closed-form solutions in
// Shkel & Lumelsky, "Classification of the Dubins set" (2001), in the
// normalised frame where the start is at the origin, the goal lies on the
// +x axis at distance d = D / rho, and headings are alpha, beta.

#include "rrt/spaces/dubins.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace rrt::dubins {
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kTwoPi = 2.0 * kPi;

inline double wrap2pi(double a) {
  // Common case first: already in range (much cheaper than fmod/floor).
  if (a >= 0.0 && a < kTwoPi) return a;
  double r = a - kTwoPi * std::floor(a / kTwoPi);
  // Rounding can land exactly on 2*pi (or a hair below 0).
  if (r >= kTwoPi || r < 0.0) r = 0.0;
  return r;
}

struct Normalised {
  double d, alpha, beta;
  double sa, sb, ca, cb, c_ab;
};

Normalised normalise(const Pose& from, const Pose& to, double rho) {
  const double dx = to.x - from.x;
  const double dy = to.y - from.y;
  const double D = std::hypot(dx, dy);
  const double phi = D > 0.0 ? std::atan2(dy, dx) : 0.0;
  Normalised n{};
  n.d = D / rho;
  n.alpha = wrap2pi(from.theta - phi);
  n.beta = wrap2pi(to.theta - phi);
  n.sa = std::sin(n.alpha);
  n.sb = std::sin(n.beta);
  n.ca = std::cos(n.alpha);
  n.cb = std::cos(n.beta);
  n.c_ab = std::cos(n.alpha - n.beta);
  return n;
}

using Params = std::array<double, 3>;

std::optional<Params> lsl(const Normalised& n) {
  const double tmp0 = n.d + n.sa - n.sb;
  const double p_sq = 2.0 + n.d * n.d - 2.0 * n.c_ab + 2.0 * n.d * (n.sa - n.sb);
  if (p_sq < 0.0) return std::nullopt;
  const double tmp1 = std::atan2(n.cb - n.ca, tmp0);
  return Params{wrap2pi(tmp1 - n.alpha), std::sqrt(p_sq), wrap2pi(n.beta - tmp1)};
}

std::optional<Params> rsr(const Normalised& n) {
  const double tmp0 = n.d - n.sa + n.sb;
  const double p_sq = 2.0 + n.d * n.d - 2.0 * n.c_ab + 2.0 * n.d * (n.sb - n.sa);
  if (p_sq < 0.0) return std::nullopt;
  const double tmp1 = std::atan2(n.ca - n.cb, tmp0);
  return Params{wrap2pi(n.alpha - tmp1), std::sqrt(p_sq), wrap2pi(tmp1 - n.beta)};
}

std::optional<Params> lsr(const Normalised& n) {
  const double p_sq =
      -2.0 + n.d * n.d + 2.0 * n.c_ab + 2.0 * n.d * (n.sa + n.sb);
  if (p_sq < 0.0) return std::nullopt;
  const double p = std::sqrt(p_sq);
  const double tmp =
      std::atan2(-n.ca - n.cb, n.d + n.sa + n.sb) - std::atan2(-2.0, p);
  return Params{wrap2pi(tmp - n.alpha), p, wrap2pi(tmp - n.beta)};
}

std::optional<Params> rsl(const Normalised& n) {
  const double p_sq =
      -2.0 + n.d * n.d + 2.0 * n.c_ab - 2.0 * n.d * (n.sa + n.sb);
  if (p_sq < 0.0) return std::nullopt;
  const double p = std::sqrt(p_sq);
  const double tmp =
      std::atan2(n.ca + n.cb, n.d - n.sa - n.sb) - std::atan2(2.0, p);
  return Params{wrap2pi(n.alpha - tmp), p, wrap2pi(n.beta - tmp)};
}

std::optional<Params> rlr(const Normalised& n) {
  const double tmp =
      (6.0 - n.d * n.d + 2.0 * n.c_ab + 2.0 * n.d * (n.sa - n.sb)) / 8.0;
  if (std::abs(tmp) > 1.0) return std::nullopt;
  const double phi = std::atan2(n.ca - n.cb, n.d - n.sa + n.sb);
  const double p = wrap2pi(kTwoPi - std::acos(tmp));
  const double t = wrap2pi(n.alpha - phi + wrap2pi(p / 2.0));
  return Params{t, p, wrap2pi(n.alpha - n.beta - t + wrap2pi(p))};
}

std::optional<Params> lrl(const Normalised& n) {
  const double tmp =
      (6.0 - n.d * n.d + 2.0 * n.c_ab + 2.0 * n.d * (n.sb - n.sa)) / 8.0;
  if (std::abs(tmp) > 1.0) return std::nullopt;
  const double phi = std::atan2(n.ca - n.cb, n.d + n.sa - n.sb);
  const double p = wrap2pi(kTwoPi - std::acos(tmp));
  const double t = wrap2pi(-n.alpha - phi + p / 2.0);
  return Params{t, p, wrap2pi(wrap2pi(n.beta) - n.alpha - t + wrap2pi(p))};
}

/// Advance a unit-radius pose along one segment of normalised length t.
Pose advance(const Pose& p, Segment seg, double t) {
  const double st = std::sin(p.theta), ct = std::cos(p.theta);
  switch (seg) {
    case Segment::kLeft:
      return {p.x + std::sin(p.theta + t) - st,
              p.y - std::cos(p.theta + t) + ct, p.theta + t};
    case Segment::kRight:
      return {p.x - std::sin(p.theta - t) + st,
              p.y + std::cos(p.theta - t) - ct, p.theta - t};
    case Segment::kStraight:
      return {p.x + t * ct, p.y + t * st, p.theta};
  }
  return p;
}

}  // namespace

double mod2pi(double a) { return wrap2pi(a); }

double wrapToPi(double a) { return mod2pi(a + kPi) - kPi; }

std::array<Segment, 3> segments(Word w) {
  using S = Segment;
  switch (w) {
    case Word::kLSL: return {S::kLeft, S::kStraight, S::kLeft};
    case Word::kLSR: return {S::kLeft, S::kStraight, S::kRight};
    case Word::kRSL: return {S::kRight, S::kStraight, S::kLeft};
    case Word::kRSR: return {S::kRight, S::kStraight, S::kRight};
    case Word::kRLR: return {S::kRight, S::kLeft, S::kRight};
    case Word::kLRL: return {S::kLeft, S::kRight, S::kLeft};
  }
  return {S::kStraight, S::kStraight, S::kStraight};
}

const char* toString(Word w) {
  switch (w) {
    case Word::kLSL: return "LSL";
    case Word::kLSR: return "LSR";
    case Word::kRSL: return "RSL";
    case Word::kRSR: return "RSR";
    case Word::kRLR: return "RLR";
    case Word::kLRL: return "LRL";
  }
  return "?";
}

namespace {

std::optional<Params> solveNormalised(const Normalised& n, Word w) {
  switch (w) {
    case Word::kLSL: return lsl(n);
    case Word::kLSR: return lsr(n);
    case Word::kRSL: return rsl(n);
    case Word::kRSR: return rsr(n);
    case Word::kRLR: return rlr(n);
    case Word::kLRL: return lrl(n);
  }
  return std::nullopt;
}

}  // namespace

std::optional<Path> solveWord(const Pose& from, const Pose& to, double rho,
                              Word w) {
  const auto params = solveNormalised(normalise(from, to, rho), w);
  if (!params) return std::nullopt;
  Path path;
  path.start = from;
  path.rho = rho;
  path.word = w;
  path.params = *params;
  return path;
}

Path shortestPath(const Pose& from, const Pose& to, double rho) {
  constexpr Word kWords[] = {Word::kLSL, Word::kLSR, Word::kRSL,
                             Word::kRSR, Word::kRLR, Word::kLRL};
  // Normalise once; this is the planner's hot path (every distance() call).
  const Normalised n = normalise(from, to, rho);
  Path best;
  best.start = from;
  best.rho = rho;
  double best_len = std::numeric_limits<double>::infinity();
  for (Word w : kWords) {
    const auto p = solveNormalised(n, w);
    if (!p) continue;
    const double len = (*p)[0] + (*p)[1] + (*p)[2];
    if (len < best_len) {
      best_len = len;
      best.word = w;
      best.params = *p;
    }
  }
  return best;
}

Pose Path::sample(double s) const {
  double t = std::clamp(s, 0.0, length()) / rho;  // normalised arc length
  const auto segs = segments(word);
  // Work in a unit-radius frame at the origin, then scale and translate.
  Pose p{0.0, 0.0, start.theta};
  for (int i = 0; i < 3 && t > 0.0; ++i) {
    const double step = std::min(t, params[i]);
    p = advance(p, segs[i], step);
    t -= step;
  }
  return {start.x + p.x * rho, start.y + p.y * rho, wrapToPi(p.theta)};
}

}  // namespace rrt::dubins
