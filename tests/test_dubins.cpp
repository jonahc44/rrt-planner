#include <gtest/gtest.h>

#include <cmath>
#include <random>

#include "rrt/spaces/dubins.hpp"
#include "rrt/spaces/dubins_state_space.hpp"

using namespace rrt;
using namespace rrt::dubins;

namespace {

constexpr double kPi = 3.14159265358979323846;

double poseError(const Pose& a, const Pose& b) {
  return std::hypot(a.x - b.x, a.y - b.y) +
         std::abs(wrapToPi(a.theta - b.theta));
}

Pose randomPose(std::mt19937_64& rng, double extent) {
  std::uniform_real_distribution<double> u(-extent, extent);
  std::uniform_real_distribution<double> ut(-kPi, kPi);
  return {u(rng), u(rng), ut(rng)};
}

}  // namespace

TEST(Dubins, EveryFeasibleWordReachesTheGoal) {
  std::mt19937_64 rng(1);
  constexpr Word kWords[] = {Word::kLSL, Word::kLSR, Word::kRSL,
                             Word::kRSR, Word::kRLR, Word::kLRL};
  int solved = 0;
  for (int i = 0; i < 2000; ++i) {
    const Pose a = randomPose(rng, 5.0), b = randomPose(rng, 5.0);
    const double rho = 0.5 + 1.5 * std::uniform_real_distribution<>(0, 1)(rng);
    for (Word w : kWords) {
      if (auto p = solveWord(a, b, rho, w)) {
        ++solved;
        EXPECT_LT(poseError(p->end(), b), 1e-6) << toString(w);
      }
    }
  }
  EXPECT_GT(solved, 2000);
}

TEST(Dubins, ShortestIsNoShorterThanStraightLine) {
  std::mt19937_64 rng(2);
  for (int i = 0; i < 2000; ++i) {
    const Pose a = randomPose(rng, 5.0), b = randomPose(rng, 5.0);
    const Path p = shortestPath(a, b, 1.0);
    EXPECT_GE(p.length() + 1e-9, std::hypot(b.x - a.x, b.y - a.y));
    EXPECT_LT(poseError(p.end(), b), 1e-6);
  }
}

TEST(Dubins, StraightAheadIsAStraightLine) {
  const Path p = shortestPath({0, 0, 0}, {3, 0, 0}, 1.0);
  EXPECT_NEAR(p.length(), 3.0, 1e-9);
}

TEST(Dubins, TurningInPlaceCostsAFullCircle) {
  // Same position, same heading -> length 0; reversed heading needs turning.
  EXPECT_NEAR(shortestPath({0, 0, 0}, {0, 0, 0}, 1.0).length(), 0.0, 1e-9);
  const Path back = shortestPath({0, 0, 0}, {0, 0, kPi}, 1.0);
  EXPECT_GT(back.length(), kPi);
  EXPECT_LT(poseError(back.end(), {0, 0, kPi}), 1e-6);
}

TEST(Dubins, SamplesStayWithinCurvatureBound) {
  // Heading change per unit arc length must never exceed 1 / rho.
  std::mt19937_64 rng(3);
  const double rho = 0.8;
  for (int i = 0; i < 200; ++i) {
    const Path p = shortestPath(randomPose(rng, 4), randomPose(rng, 4), rho);
    const double ds = 0.01;
    Pose prev = p.sample(0.0);
    for (double s = ds; s <= p.length(); s += ds) {
      const Pose cur = p.sample(s);
      EXPECT_LE(std::abs(wrapToPi(cur.theta - prev.theta)), ds / rho + 1e-6);
      prev = cur;
    }
  }
}

TEST(DubinsStateSpace, EmbeddingIsALowerBound) {
  DubinsStateSpace space(1.0, -5, 5, -5, 5);
  Rng rng(4);
  for (int i = 0; i < 1000; ++i) {
    const State a = space.sampleUniform(rng), b = space.sampleUniform(rng);
    double ea[2], eb[2];
    space.embed(a, ea);
    space.embed(b, eb);
    const double e = std::hypot(ea[0] - eb[0], ea[1] - eb[1]);
    EXPECT_LE(e, space.distance(a, b) + 1e-9);
    EXPECT_LE(e, space.distance(b, a) + 1e-9);
  }
}

TEST(DubinsStateSpace, SteerTruncatesAndEndsOnThePath) {
  DubinsStateSpace space(1.0, -5, 5, -5, 5);
  const State a{0, 0, 0}, b{4, 4, kPi / 2};
  const auto full = space.connect(a, b);
  const auto part = space.steer(a, b, 1.0);
  EXPECT_NEAR(part->length(), 1.0, 1e-9);
  const State e = part->end(), f = full->interpolate(1.0);
  for (int k = 0; k < 3; ++k) EXPECT_NEAR(e[k], f[k], 1e-9);
}

TEST(Dubins, NoSpuriousFullLoopsOnAlignedEndpoints) {
  // Regression: a segment whose turn should be exactly zero came out as
  // -1e-16 and wrapped to 2*pi, adding a full loop (2*pi*rho) to the
  // length. It hit most connections to truncated-steer endpoints.
  std::mt19937_64 rng(12);
  const double rho = 0.8;
  int checked = 0;
  for (int i = 0; i < 2000; ++i) {
    const Pose a = randomPose(rng, 8.0), b = randomPose(rng, 8.0);
    const Path full = shortestPath(a, b, rho);
    for (double s : {0.1, 0.5, 1.0, 0.5 * full.length()}) {
      if (s > full.length()) continue;
      const Pose mid = full.sample(s);
      EXPECT_NEAR(shortestPath(a, mid, rho).length(), s, 1e-6);
      ++checked;
    }
  }
  EXPECT_GT(checked, 7000);
  // The two textbook aligned cases.
  EXPECT_NEAR(shortestPath({0, 0, 0}, {5, 0, 0}, rho).length(), 5.0, 1e-12);
  EXPECT_NEAR(shortestPath({0, 0, kPi / 2}, {0, 5, kPi / 2}, rho).length(), 5.0, 1e-12);
}
