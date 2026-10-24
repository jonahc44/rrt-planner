// Steering functions: StateSpace::connect / steer and the LocalPaths they
// return, for every bundled space. These are the contracts RRTStar relies on.

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <memory>
#include <vector>

#include "rrt/core/local_path.hpp"
#include "rrt/core/state_space.hpp"
#include "rrt/spaces/dubins_state_space.hpp"
#include "rrt/spaces/real_vector_2d_space.hpp"

using namespace rrt;

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kTol = 1e-9;

double angleDiff(double a, double b) { return std::abs(dubins::wrapToPi(a - b)); }

void expectStateNear(const State& a, const State& b, double tol = 1e-7) {
  ASSERT_EQ(a.size(), b.size());
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (i == 2) {
      EXPECT_LT(angleDiff(a[i], b[i]), tol) << "heading";
    } else {
      EXPECT_NEAR(a[i], b[i], tol) << "component " << i;
    }
  }
}

/// Arc length actually travelled along a path, measured by fine sampling.
double measuredLength(const LocalPath& p, int samples = 4000) {
  double total = 0.0;
  State prev = p.interpolate(0.0);
  for (int i = 1; i <= samples; ++i) {
    const State cur = p.interpolate(p.length() * i / samples);
    total += std::hypot(cur[0] - prev[0], cur[1] - prev[1]);
    prev = cur;
  }
  return total;
}

/// Contracts every StateSpace must satisfy, checked over random pairs.
void checkSteeringContract(const StateSpace& space, double max_step) {
  Rng rng(11);
  for (int i = 0; i < 300; ++i) {
    const State a = space.sampleUniform(rng), b = space.sampleUniform(rng);
    const LocalPathPtr full = space.connect(a, b);
    ASSERT_TRUE(full);
    // connect() is exact at both ends.
    expectStateNear(full->start(), a);
    expectStateNear(full->end(), b);
    // distance() is the length of connect()'s path. RRTStar filters
    // neighbours by edge length on the strength of this.
    EXPECT_NEAR(space.distance(a, b), full->length(), kTol);

    const LocalPathPtr part = space.steer(a, b, max_step);
    ASSERT_TRUE(part);
    EXPECT_LE(part->length(), max_step + kTol);
    EXPECT_NEAR(part->length(), std::min(max_step, full->length()), kTol);
    expectStateNear(part->start(), a);
    // The truncated path is a prefix of the full one.
    for (double s : {0.0, 0.3 * part->length(), part->length()}) {
      expectStateNear(part->interpolate(s), full->interpolate(s));
    }
    // If the target is within reach, steer() gets there exactly.
    if (full->length() <= max_step) expectStateNear(part->end(), b);
  }
}

}  // namespace

// ---- R^2 --------------------------------------------------------------------

TEST(StraightLineSteering, InterpolatesLinearlyByArcLength) {
  RealVector2DSpace space(0, 10, 0, 10);
  const auto p = space.connect({1, 1}, {4, 5});
  EXPECT_NEAR(p->length(), 5.0, kTol);
  expectStateNear(p->interpolate(2.5), {2.5, 3.0});
  expectStateNear(p->interpolate(1.0), {1.6, 1.8});
}

TEST(StraightLineSteering, InterpolationClampsOutsideTheArc) {
  RealVector2DSpace space(0, 10, 0, 10);
  const auto p = space.connect({0, 0}, {3, 4});
  expectStateNear(p->interpolate(-1.0), {0, 0});
  expectStateNear(p->interpolate(99.0), {3, 4});
}

TEST(StraightLineSteering, ZeroLengthPathIsWellDefined) {
  RealVector2DSpace space(0, 10, 0, 10);
  const auto p = space.connect({2, 2}, {2, 2});
  EXPECT_EQ(p->length(), 0.0);
  expectStateNear(p->interpolate(0.0), {2, 2});
  expectStateNear(p->interpolate(1.0), {2, 2});  // no division by zero
  EXPECT_EQ(space.steer({2, 2}, {2, 2}, 0.5)->length(), 0.0);
}

TEST(StraightLineSteering, SteerStopsAtMaxStepAlongTheLine) {
  RealVector2DSpace space(0, 10, 0, 10);
  const auto p = space.steer({0, 0}, {6, 8}, 2.0);
  EXPECT_NEAR(p->length(), 2.0, kTol);
  expectStateNear(p->end(), {1.2, 1.6});
}

TEST(StraightLineSteering, SteerWithinReachReturnsTheFullPath) {
  RealVector2DSpace space(0, 10, 0, 10);
  const auto p = space.steer({0, 0}, {0.3, 0.4}, 2.0);
  EXPECT_NEAR(p->length(), 0.5, kTol);
  expectStateNear(p->end(), {0.3, 0.4});
}

TEST(StraightLineSteering, DistanceIsSymmetricEuclidean) {
  RealVector2DSpace space(0, 10, 0, 10);
  EXPECT_NEAR(space.distance({1, 2}, {4, 6}), 5.0, kTol);
  EXPECT_NEAR(space.distance({4, 6}, {1, 2}), 5.0, kTol);
}

TEST(StraightLineSteering, SatisfiesTheStateSpaceContract) {
  checkSteeringContract(RealVector2DSpace(0, 10, 0, 10), 0.75);
}

// ---- TruncatedPath ------------------------------------------------------------

TEST(TruncatedPath, ClampsLengthToTheInnerPath) {
  auto inner = std::make_shared<StraightLinePath>(State{0, 0}, State{3, 4});
  EXPECT_NEAR(TruncatedPath(inner, 2.0).length(), 2.0, kTol);
  EXPECT_NEAR(TruncatedPath(inner, 10.0).length(), 5.0, kTol);
  EXPECT_EQ(TruncatedPath(inner, -1.0).length(), 0.0);
}

TEST(TruncatedPath, NeverInterpolatesPastTheCut) {
  auto inner = std::make_shared<StraightLinePath>(State{0, 0}, State{10, 0});
  TruncatedPath p(inner, 4.0);
  expectStateNear(p.interpolate(7.0), {4, 0});
  expectStateNear(p.end(), {4, 0});
  expectStateNear(p.interpolate(-3.0), {0, 0});
}

// ---- Dubins -----------------------------------------------------------------

class DubinsSteering : public ::testing::Test {
 protected:
  static constexpr double kRho = 0.8;
  DubinsStateSpace space{kRho, -10, 10, -10, 10};
};

TEST_F(DubinsSteering, SatisfiesTheStateSpaceContract) {
  checkSteeringContract(space, 1.0);
}

TEST_F(DubinsSteering, ParameterisationIsArcLength) {
  Rng rng(5);
  for (int i = 0; i < 50; ++i) {
    const auto p = space.connect(space.sampleUniform(rng), space.sampleUniform(rng));
    EXPECT_NEAR(measuredLength(*p), p->length(), 1e-3 * std::max(1.0, p->length()));
  }
}

TEST_F(DubinsSteering, HeadingIsTangentToTheTrajectory) {
  // A forward-only car moves along its heading; it never slides sideways.
  Rng rng(6);
  const double ds = 1e-4;
  for (int i = 0; i < 50; ++i) {
    const auto p = space.connect(space.sampleUniform(rng), space.sampleUniform(rng));
    for (double s = 0.0; s + ds <= p->length(); s += p->length() / 20) {
      const State a = p->interpolate(s), b = p->interpolate(s + ds);
      const double travel = std::atan2(b[1] - a[1], b[0] - a[0]);
      EXPECT_LT(angleDiff(travel, a[2]), 1e-3);
    }
  }
}

TEST_F(DubinsSteering, HeadingsAreWrappedToMinusPiPi) {
  Rng rng(7);
  for (int i = 0; i < 50; ++i) {
    const auto p = space.connect(space.sampleUniform(rng), space.sampleUniform(rng));
    for (double s = 0.0; s <= p->length(); s += 0.05) {
      const double th = p->interpolate(s)[2];
      EXPECT_GE(th, -kPi);
      EXPECT_LT(th, kPi);
    }
  }
}

TEST_F(DubinsSteering, DistanceIsAsymmetric) {
  // Directly ahead is a straight line; directly behind needs a turn-around.
  const State a{0, 0, 0}, b{2, 0, 0};
  EXPECT_NEAR(space.distance(a, b), 2.0, kTol);
  EXPECT_GT(space.distance(b, a), 2.0 + kPi * kRho);
}

TEST_F(DubinsSteering, LateralNeighbourIsFarDespiteSmallEuclideanGap) {
  // Sideways by 0.1 with the same heading: Euclidean 0.1, but the car must
  // swerve, which costs far more than 0.1.
  EXPECT_GT(space.distance({0, 0, 0}, {0, 0.1, 0}), 1.0);
}

TEST_F(DubinsSteering, ShortestPathScalesWithTurningRadius) {
  // In the normalised frame the solution is scale-free: scaling positions
  // and rho together scales the length.
  DubinsStateSpace big(2 * kRho, -20, 20, -20, 20);
  Rng rng(8);
  for (int i = 0; i < 100; ++i) {
    const State a = space.sampleUniform(rng), b = space.sampleUniform(rng);
    const State a2{2 * a[0], 2 * a[1], a[2]}, b2{2 * b[0], 2 * b[1], b[2]};
    EXPECT_NEAR(big.distance(a2, b2), 2 * space.distance(a, b), 1e-7);
  }
}

TEST_F(DubinsSteering, TriangleInequalityHolds) {
  // An optimal steering cost can't beat going via an intermediate state.
  Rng rng(9);
  for (int i = 0; i < 500; ++i) {
    const State a = space.sampleUniform(rng), b = space.sampleUniform(rng),
                c = space.sampleUniform(rng);
    EXPECT_LE(space.distance(a, c),
              space.distance(a, b) + space.distance(b, c) + 1e-7);
  }
}

TEST_F(DubinsSteering, TruncatedSteerIsItselfOptimal) {
  // Any prefix of a shortest Dubins path is a shortest path to its end
  // point, so steer() never produces a detour.
  Rng rng(10);
  for (int i = 0; i < 200; ++i) {
    const State a = space.sampleUniform(rng), b = space.sampleUniform(rng);
    const auto part = space.steer(a, b, 1.0);
    EXPECT_NEAR(space.distance(a, part->end()), part->length(), 1e-6);
  }
}

TEST_F(DubinsSteering, SameStateIsZeroLength) {
  const State a{1, 2, 0.3};
  EXPECT_NEAR(space.distance(a, a), 0.0, 1e-12);
  expectStateNear(space.steer(a, a, 1.0)->end(), a);
}

TEST_F(DubinsSteering, HandlesHeadingsOutsideTheCanonicalRange) {
  // Callers may hand in unwrapped headings; they denote the same pose.
  const State a{0, 0, 0.5}, b{3, 1, -2.0};
  const State a_wrapped{0, 0, 0.5 + 2 * kPi}, b_wrapped{3, 1, -2.0 - 4 * kPi};
  EXPECT_NEAR(space.distance(a, b), space.distance(a_wrapped, b_wrapped), 1e-9);
}

TEST_F(DubinsSteering, EmbeddingIsXY) {
  double e[2];
  space.embed({1.5, -2.5, 1.0}, e);
  EXPECT_EQ(e[0], 1.5);
  EXPECT_EQ(e[1], -2.5);
  EXPECT_EQ(space.embeddingDimension(), 2u);
}

// ---- Default steer() on a user-defined space ---------------------------------

namespace {

/// A space whose connect() fails for targets with negative x, to check
/// that the default steer() propagates "no motion".
class OneSidedSpace final : public StateSpace {
 public:
  std::size_t dimension() const override { return 2; }
  State sampleUniform(Rng&) const override { return {0, 0}; }
  double distance(const State& a, const State& b) const override {
    return b[0] < 0 ? std::numeric_limits<double>::infinity()
                    : std::hypot(b[0] - a[0], b[1] - a[1]);
  }
  LocalPathPtr connect(const State& a, const State& b) const override {
    if (b[0] < 0) return nullptr;
    return std::make_shared<StraightLinePath>(a, b);
  }
  std::size_t embeddingDimension() const override { return 2; }
  void embed(const State& s, double* out) const override {
    out[0] = s[0];
    out[1] = s[1];
  }
  double measure() const override { return 1.0; }
};

}  // namespace

TEST(DefaultSteer, PropagatesInfeasibleConnections) {
  OneSidedSpace space;
  EXPECT_EQ(space.steer({0, 0}, {-1, 0}, 1.0), nullptr);
  EXPECT_NEAR(space.steer({0, 0}, {4, 0}, 1.0)->length(), 1.0, kTol);
}

TEST(DefaultSteer, DoesNotWrapPathsThatAreAlreadyShortEnough) {
  OneSidedSpace space;
  const auto p = space.steer({0, 0}, {0.5, 0}, 1.0);
  EXPECT_NE(dynamic_cast<const StraightLinePath*>(p.get()), nullptr);
}
