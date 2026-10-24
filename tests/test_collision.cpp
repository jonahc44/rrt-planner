// Collision checking: GeometricWorld2D, OccupancyGrid + GridCollisionChecker,
// and the default CollisionChecker::isMotionValid sampling.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "rrt/collision/geometric_world_2d.hpp"
#include "rrt/collision/occupancy_grid.hpp"
#include "rrt/core/collision_checker.hpp"
#include "rrt/spaces/dubins_state_space.hpp"
#include "rrt/spaces/real_vector_2d_space.hpp"

using namespace rrt;

namespace {

constexpr double kPi = 3.14159265358979323846;

LocalPathPtr line(State a, State b) {
  return std::make_shared<StraightLinePath>(std::move(a), std::move(b));
}

/// Records every state it is asked about; obstacle-free.
class RecordingChecker final : public CollisionChecker {
 public:
  explicit RecordingChecker(double resolution) : CollisionChecker(resolution) {}
  bool isStateValid(const State& s) const override {
    queried.push_back(s);
    return true;
  }
  mutable std::vector<State> queried;
};

}  // namespace

// ---- GeometricWorld2D: states ------------------------------------------------

TEST(GeometricWorld2D, EmptyWorldAcceptsInteriorPoints) {
  GeometricWorld2D w(0, 10, 0, 5);
  EXPECT_TRUE(w.isStateValid({5, 2.5}));
  EXPECT_TRUE(w.isStateValid({0, 0}));    // boundary is inclusive for a point robot
  EXPECT_TRUE(w.isStateValid({10, 5}));
}

TEST(GeometricWorld2D, RejectsPointsOutsideTheBounds) {
  GeometricWorld2D w(0, 10, 0, 5);
  EXPECT_FALSE(w.isStateValid({-0.01, 2}));
  EXPECT_FALSE(w.isStateValid({10.01, 2}));
  EXPECT_FALSE(w.isStateValid({5, -0.01}));
  EXPECT_FALSE(w.isStateValid({5, 5.01}));
}

TEST(GeometricWorld2D, RobotRadiusShrinksTheFreeBounds) {
  GeometricWorld2D w(0, 10, 0, 10, /*robot_radius=*/0.5);
  EXPECT_FALSE(w.isStateValid({0.4, 5}));
  EXPECT_TRUE(w.isStateValid({0.5, 5}));
  EXPECT_FALSE(w.isStateValid({5, 9.6}));
  EXPECT_TRUE(w.isStateValid({5, 9.5}));
}

TEST(GeometricWorld2D, CircleObstacle) {
  GeometricWorld2D w(0, 10, 0, 10);
  w.addCircle({5, 5, 1});
  EXPECT_FALSE(w.isStateValid({5, 5}));
  EXPECT_FALSE(w.isStateValid({5.7, 5.7}));  // inside: 0.99 from centre
  EXPECT_FALSE(w.isStateValid({6, 5}));      // touching counts as contact
  EXPECT_TRUE(w.isStateValid({6.001, 5}));
  EXPECT_TRUE(w.isStateValid({5.8, 5.8}));   // outside: 1.13 from centre
}

TEST(GeometricWorld2D, CircleIsInflatedByRobotRadius) {
  GeometricWorld2D w(0, 10, 0, 10, /*robot_radius=*/0.3);
  w.addCircle({5, 5, 1});
  EXPECT_FALSE(w.isStateValid({6.25, 5}));
  EXPECT_TRUE(w.isStateValid({6.31, 5}));
}

TEST(GeometricWorld2D, BoxObstacle) {
  GeometricWorld2D w(0, 10, 0, 10);
  w.addBox({2, 3, 4, 6});
  EXPECT_FALSE(w.isStateValid({3, 4.5}));  // inside
  EXPECT_FALSE(w.isStateValid({2, 4}));    // on an edge
  EXPECT_FALSE(w.isStateValid({4, 6}));    // on a corner
  EXPECT_TRUE(w.isStateValid({1.99, 4}));
  EXPECT_TRUE(w.isStateValid({3, 6.01}));
}

TEST(GeometricWorld2D, BoxCornersAreRoundedByRobotRadius) {
  // A disc grazing a box corner is tested against the true (rounded)
  // Minkowski sum, not an inflated square.
  GeometricWorld2D w(0, 10, 0, 10, /*robot_radius=*/0.5);
  w.addBox({2, 2, 4, 4});
  EXPECT_FALSE(w.isStateValid({4.45, 3}));          // 0.45 from the right face
  EXPECT_TRUE(w.isStateValid({4.51, 3}));
  // Diagonally off the corner: 0.4 along each axis = 0.566 > 0.5 away.
  EXPECT_TRUE(w.isStateValid({4.4, 4.4}));
  // An inflated-square test would have rejected that point.
  EXPECT_FALSE(w.isStateValid({4.3, 4.3}));         // 0.42 away
}

TEST(GeometricWorld2D, IgnoresHeadingForADiscRobot) {
  GeometricWorld2D w(0, 10, 0, 10, 0.2);
  w.addCircle({5, 5, 1});
  for (double th : {-3.0, 0.0, 1.0, 3.0}) {
    EXPECT_TRUE(w.isStateValid({2, 2, th}));
    EXPECT_FALSE(w.isStateValid({5, 5, th}));
  }
}

TEST(GeometricWorld2D, ExposesItsGeometry) {
  GeometricWorld2D w(-1, 2, -3, 4);
  w.addCircle({0, 0, 1});
  w.addBox({0, 0, 1, 1});
  w.addBox({1, 1, 2, 2});
  EXPECT_EQ(w.circles().size(), 1u);
  EXPECT_EQ(w.boxes().size(), 2u);
  EXPECT_EQ(w.xMin(), -1);
  EXPECT_EQ(w.yMax(), 4);
}

// ---- Motion validity ---------------------------------------------------------

TEST(MotionValidity, DetectsObstacleInTheMiddleOfAnEdge) {
  GeometricWorld2D w(0, 10, 0, 10, 0.0, 0.05);
  w.addBox({4.9, 0, 5.1, 10});
  EXPECT_TRUE(w.isStateValid({1, 5}));
  EXPECT_TRUE(w.isStateValid({9, 5}));
  EXPECT_FALSE(w.isMotionValid(*line({1, 5}, {9, 5})));
  EXPECT_TRUE(w.isMotionValid(*line({1, 5}, {4, 5})));
}

TEST(MotionValidity, RejectsMotionsEndingInCollision) {
  GeometricWorld2D w(0, 10, 0, 10);
  w.addCircle({8, 5, 0.5});
  EXPECT_FALSE(w.isMotionValid(*line({1, 5}, {8, 5})));
}

TEST(MotionValidity, RejectsMotionsStartingInCollision) {
  GeometricWorld2D w(0, 10, 0, 10);
  w.addCircle({1, 5, 0.5});
  EXPECT_FALSE(w.isMotionValid(*line({1, 5}, {8, 5})));
}

TEST(MotionValidity, RejectsMotionsLeavingTheWorld) {
  GeometricWorld2D w(0, 10, 0, 10);
  // A Dubins turn-around near the wall swings outside the map.
  DubinsStateSpace space(1.0, 0, 10, 0, 10);
  const auto p = space.connect({9.5, 5, 0}, {9.5, 4, kPi});
  EXPECT_FALSE(w.isMotionValid(*p));
}

TEST(MotionValidity, SamplesNoCoarserThanTheResolution) {
  for (double res : {0.5, 0.1, 0.013}) {
    RecordingChecker c(res);
    const auto p = line({0, 0}, {3.7, 0});
    ASSERT_TRUE(c.isMotionValid(*p));
    // Every point on the segment is within res/2 of a checked state, and
    // both endpoints are checked exactly.
    std::vector<double> xs;
    for (const State& s : c.queried) xs.push_back(s[0]);
    std::sort(xs.begin(), xs.end());
    EXPECT_DOUBLE_EQ(xs.front(), 0.0);
    EXPECT_DOUBLE_EQ(xs.back(), 3.7);
    for (std::size_t i = 1; i < xs.size(); ++i) {
      EXPECT_LE(xs[i] - xs[i - 1], res + 1e-12) << "res=" << res;
    }
    // ...and without wasted work: about length/res checks, not many more.
    EXPECT_LE(c.queried.size(), std::ceil(3.7 / res) + 2);
  }
}

TEST(MotionValidity, ZeroLengthMotionChecksTheState) {
  RecordingChecker c(0.1);
  EXPECT_TRUE(c.isMotionValid(*line({2, 3}, {2, 3})));
  EXPECT_GE(c.queried.size(), 1u);

  GeometricWorld2D w(0, 10, 0, 10);
  w.addCircle({2, 3, 0.5});
  EXPECT_FALSE(w.isMotionValid(*line({2, 3}, {2, 3})));
}

TEST(MotionValidity, ThinObstaclesNeedAFineEnoughResolution) {
  // Documents the sampling trade-off: an obstacle thinner than the
  // resolution can be stepped over. Pick resolution < thinnest obstacle.
  GeometricWorld2D coarse(0, 10, 0, 10, 0.0, /*resolution=*/1.0);
  GeometricWorld2D fine(0, 10, 0, 10, 0.0, /*resolution=*/0.02);
  for (GeometricWorld2D* w : {&coarse, &fine}) w->addBox({4.95, 0, 5.05, 10});
  const auto p = line({0.5, 5}, {9.5, 5});  // samples at 0.5, 1.5, ... skip 5.0
  EXPECT_TRUE(coarse.isMotionValid(*p));
  EXPECT_FALSE(fine.isMotionValid(*p));
}

TEST(MotionValidity, ResolutionIsAdjustable) {
  GeometricWorld2D w(0, 10, 0, 10, 0.0, 1.0);
  w.addBox({4.95, 0, 5.05, 10});
  const auto p = line({0.5, 5}, {9.5, 5});
  EXPECT_TRUE(w.isMotionValid(*p));
  w.setResolution(0.05);
  EXPECT_DOUBLE_EQ(w.resolution(), 0.05);
  EXPECT_FALSE(w.isMotionValid(*p));
}

TEST(MotionValidity, FollowsCurvedPaths) {
  // A circle sitting off the chord of a Dubins arc: the straight segment
  // between the endpoints misses it, the curve does not.
  DubinsStateSpace space(1.0, -10, 10, -10, 10);
  const auto arc = space.connect({0, 0, 0}, {0, 2, kPi});  // U-turn left
  GeometricWorld2D w(-10, 10, -10, 10);
  w.addCircle({1.0, 1.0, 0.1});  // on the arc's apex
  EXPECT_FALSE(w.isMotionValid(*arc));
  EXPECT_TRUE(w.isMotionValid(*line({0, 0}, {0, 2})));
}

// ---- OccupancyGrid -----------------------------------------------------------

TEST(OccupancyGrid, AsciiFirstLineIsTheTopRow) {
  const auto g = OccupancyGrid::fromAscii({"#..",
                                           "...",
                                           "..9"},
                                          1.0);
  EXPECT_EQ(g.width(), 3u);
  EXPECT_EQ(g.height(), 3u);
  EXPECT_DOUBLE_EQ(g.at(0, 2), 1.0);  // top-left
  EXPECT_DOUBLE_EQ(g.at(2, 0), 1.0);  // bottom-right, '9' = 9/9
  EXPECT_DOUBLE_EQ(g.valueAt(0.5, 2.5), 1.0);
  EXPECT_DOUBLE_EQ(g.valueAt(1.5, 1.5), 0.0);
}

TEST(OccupancyGrid, DigitsMapToFractions) {
  const auto g = OccupancyGrid::fromAscii({"0369"}, 1.0);
  EXPECT_DOUBLE_EQ(g.valueAt(0.5, 0.5), 0.0);
  EXPECT_DOUBLE_EQ(g.valueAt(1.5, 0.5), 3.0 / 9.0);
  EXPECT_DOUBLE_EQ(g.valueAt(2.5, 0.5), 6.0 / 9.0);
  EXPECT_DOUBLE_EQ(g.valueAt(3.5, 0.5), 1.0);
}

TEST(OccupancyGrid, RejectsMalformedAscii) {
  EXPECT_THROW(OccupancyGrid::fromAscii({}, 1.0), std::invalid_argument);
  EXPECT_THROW(OccupancyGrid::fromAscii({"..", "..."}, 1.0), std::invalid_argument);
  EXPECT_THROW(OccupancyGrid::fromAscii({".x"}, 1.0), std::invalid_argument);
}

TEST(OccupancyGrid, RespectsOriginAndResolution) {
  OccupancyGrid g(4, 2, 0.5, /*origin_x=*/-1.0, /*origin_y=*/2.0);
  g.at(3, 1) = 0.7;  // covers x in [0.5, 1.0), y in [2.5, 3.0)
  EXPECT_DOUBLE_EQ(g.valueAt(0.75, 2.75), 0.7);
  EXPECT_DOUBLE_EQ(g.valueAt(0.25, 2.75), 0.0);
  EXPECT_TRUE(g.inBounds(-1.0, 2.0));    // lower edges inclusive
  EXPECT_FALSE(g.inBounds(1.0, 2.5));    // upper edges exclusive
  EXPECT_FALSE(g.inBounds(0.0, 3.0));
  EXPECT_FALSE(g.inBounds(-1.01, 2.5));
}

TEST(OccupancyGrid, OutsideValueIsConfigurable) {
  OccupancyGrid g(2, 2, 1.0);
  EXPECT_DOUBLE_EQ(g.valueAt(-1, 0), 1.0);  // unknown = occupied by default
  EXPECT_DOUBLE_EQ(g.valueAt(-1, 0, 0.25), 0.25);
  EXPECT_DOUBLE_EQ(g.valueAt(5, 5, 0.0), 0.0);
}

TEST(OccupancyGrid, CellBoundariesBelongToTheUpperCell) {
  OccupancyGrid g(3, 1, 1.0);
  g.at(1, 0) = 1.0;
  EXPECT_DOUBLE_EQ(g.valueAt(0.999, 0.5), 0.0);
  EXPECT_DOUBLE_EQ(g.valueAt(1.0, 0.5), 1.0);
  EXPECT_DOUBLE_EQ(g.valueAt(2.0, 0.5), 0.0);
}

// ---- GridCollisionChecker ------------------------------------------------------

TEST(GridCollisionChecker, ThresholdSeparatesFreeFromOccupied) {
  auto grid = std::make_shared<OccupancyGrid>(
      OccupancyGrid::fromAscii({"0459#"}, 1.0));
  GridCollisionChecker strict(grid, /*occupied_threshold=*/0.5);
  GridCollisionChecker lenient(grid, /*occupied_threshold=*/0.99);
  EXPECT_TRUE(strict.isStateValid({0.5, 0.5}));
  EXPECT_TRUE(strict.isStateValid({1.5, 0.5}));   // 4/9 = 0.44
  EXPECT_FALSE(strict.isStateValid({2.5, 0.5}));  // 5/9 = 0.56
  EXPECT_TRUE(lenient.isStateValid({2.5, 0.5}));
  EXPECT_FALSE(lenient.isStateValid({4.5, 0.5}));
}

TEST(GridCollisionChecker, OutsideTheMapIsOccupied) {
  auto grid = std::make_shared<OccupancyGrid>(2, 2, 1.0);
  GridCollisionChecker c(grid);
  EXPECT_TRUE(c.isStateValid({1, 1}));
  EXPECT_FALSE(c.isStateValid({-0.1, 1}));
  EXPECT_FALSE(c.isStateValid({1, 2.0}));
}

TEST(GridCollisionChecker, MotionThroughAWallIsRejected) {
  auto grid = std::make_shared<OccupancyGrid>(OccupancyGrid::fromAscii(
      {"..#..",
       "..#..",
       "....."},
      1.0));
  GridCollisionChecker c(grid, 0.5, 0.1);
  EXPECT_FALSE(c.isMotionValid(*line({0.5, 2.5}, {4.5, 2.5})));  // through '#'
  EXPECT_TRUE(c.isMotionValid(*line({0.5, 0.5}, {4.5, 0.5})));   // under it
}

TEST(GridCollisionChecker, AgreesWithGeometricWorldOnAxisAlignedBoxes) {
  // The same box as a geometric primitive and rasterised on a 0.1 grid.
  GeometricWorld2D geo(0, 5, 0, 5);
  geo.addBox({1.0, 1.0, 3.0, 2.0});
  auto grid = std::make_shared<OccupancyGrid>(50, 50, 0.1);
  for (std::size_t r = 10; r < 20; ++r)
    for (std::size_t c = 10; c < 30; ++c) grid->at(c, r) = 1.0;
  GridCollisionChecker raster(grid);
  Rng rng(3);
  std::uniform_real_distribution<double> u(0.0, 4.999);
  int checked = 0;
  for (int i = 0; i < 5000; ++i) {
    const State s{u(rng), u(rng)};
    // Skip points within a cell of the box outline, where rasterisation
    // and the exact test may legitimately differ.
    const bool near_edge = std::abs(s[0] - 1.0) < 0.1 || std::abs(s[0] - 3.0) < 0.1 ||
                           std::abs(s[1] - 1.0) < 0.1 || std::abs(s[1] - 2.0) < 0.1;
    if (near_edge) continue;
    ++checked;
    EXPECT_EQ(geo.isStateValid(s), raster.isStateValid(s)) << s[0] << ", " << s[1];
  }
  EXPECT_GT(checked, 4000);
}
