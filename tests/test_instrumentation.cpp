#include <gtest/gtest.h>

#include <memory>

#include "rrt/collision/geometric_world_2d.hpp"
#include "rrt/core/instrumentation.hpp"
#include "rrt/planners/rrt_star.hpp"
#include "rrt/spaces/real_vector_2d_space.hpp"

using namespace rrt;

TEST(Instrumentation, CountersForwardAndCount) {
  auto inner = std::make_shared<RealVector2DSpace>(0, 10, 0, 10);
  CountingStateSpace space(inner);
  EXPECT_EQ(space.distance({0, 0}, {3, 4}), 5.0);
  EXPECT_EQ(space.connect({0, 0}, {3, 4})->length(), 5.0);
  EXPECT_NEAR(space.steer({0, 0}, {3, 4}, 1.0)->length(), 1.0, 1e-12);
  EXPECT_EQ(space.distanceCalls(), 1u);
  EXPECT_EQ(space.connectCalls(), 1u);
  EXPECT_EQ(space.steerCalls(), 1u);
  space.reset();
  EXPECT_EQ(space.distanceCalls(), 0u);

  auto world = std::make_shared<GeometricWorld2D>(0, 10, 0, 10, 0.0, 0.1);
  world->addCircle({5, 5, 1});
  CountingCollisionChecker checker(world);
  EXPECT_DOUBLE_EQ(checker.resolution(), 0.1);
  EXPECT_FALSE(checker.isStateValid({5, 5}));
  EXPECT_FALSE(checker.isMotionValid(StraightLinePath({0, 5}, {10, 5})));
  EXPECT_EQ(checker.stateCalls(), 1u);
  EXPECT_EQ(checker.motionCalls(), 1u);
  EXPECT_NEAR(checker.checkedLength(), 10.0, 1e-5);
}

TEST(Instrumentation, PlannerWorkIsDeterministicForAFixedSeed) {
  const auto run = [] {
    auto space = std::make_shared<CountingStateSpace>(
        std::make_shared<RealVector2DSpace>(0, 10, 0, 10));
    auto world = std::make_shared<GeometricWorld2D>(0, 10, 0, 10, 0.1, 0.05);
    world->addBox({4.5, 0, 5.5, 7});
    auto checker = std::make_shared<CountingCollisionChecker>(world);
    PlanningProblem p{space, checker, std::make_shared<PathLengthCost>(),
                      {1, 1}, std::make_shared<GoalState>(space, State{9, 9}, 0.1)};
    RRTStar planner;
    planner.setProblem(p);
    planner.solve({2000, 0.0, false});
    return std::make_tuple(space->connectCalls(), space->distanceCalls(),
                           checker->motionCalls(), checker->checkedLength());
  };
  const auto a = run(), b = run();
  EXPECT_EQ(a, b);
  EXPECT_GT(std::get<0>(a), 2000u);
}
