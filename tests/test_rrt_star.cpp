#include <gtest/gtest.h>

#include <cmath>

#include "rrt/collision/geometric_world_2d.hpp"
#include "rrt/collision/occupancy_grid.hpp"
#include "rrt/cost/costmap_cost.hpp"
#include "rrt/planners/rrt_star.hpp"
#include "rrt/spaces/dubins_state_space.hpp"
#include "rrt/spaces/real_vector_2d_space.hpp"

using namespace rrt;

namespace {

constexpr double kPi = 3.14159265358979323846;

PlanningProblem r2Problem(std::shared_ptr<CollisionChecker> checker,
                          std::shared_ptr<CostFunction> cost = nullptr) {
  auto space = std::make_shared<RealVector2DSpace>(0, 10, 0, 10);
  PlanningProblem p;
  p.space = space;
  p.checker = std::move(checker);
  p.cost = cost ? std::move(cost) : std::make_shared<PathLengthCost>();
  p.start = {1, 1};
  p.goal = std::make_shared<GoalState>(space, State{9, 9}, 0.1);
  return p;
}

std::shared_ptr<GeometricWorld2D> wallWorld(double robot_radius) {
  auto w = std::make_shared<GeometricWorld2D>(0, 10, 0, 10, robot_radius, 0.02);
  w->addBox({4.5, 0.0, 5.5, 7.0});
  return w;
}

double distanceBetween(const State& a, const State& b) {
  double sq = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    double d = a[i] - b[i];
    if (i == 2) d = dubins::wrapToPi(d);
    sq += d * d;
  }
  return std::sqrt(sq);
}

/// Structural invariants that must hold for any space / cost combination.
void checkTreeInvariants(const RRTStar& planner, const CostFunction& cost) {
  const auto& tree = planner.tree();
  for (std::size_t i = 1; i < tree.size(); ++i) {
    const auto& n = tree[i];
    ASSERT_NE(n.parent, RRTStar::kNoParent);
    ASSERT_TRUE(n.edge);
    const auto& parent = tree[n.parent];
    EXPECT_NEAR(n.cost, parent.cost + n.edge_cost, 1e-6);
    EXPECT_NEAR(n.edge_cost, cost.motionCost(*n.edge), 1e-6);
    EXPECT_LT(distanceBetween(n.edge->start(), parent.state), 1e-6);
    EXPECT_LT(distanceBetween(n.edge->end(), n.state), 1e-6);
  }
}

void checkSolution(const PlannerResult& r, const PlanningProblem& p) {
  ASSERT_TRUE(r.solved());
  ASSERT_EQ(r.segments.size() + 1, r.waypoints.size());
  EXPECT_LT(distanceBetween(r.waypoints.front(), p.start), 1e-9);
  EXPECT_TRUE(p.goal->isSatisfied(r.waypoints.back()));
  for (std::size_t i = 0; i < r.segments.size(); ++i) {
    EXPECT_TRUE(p.checker->isMotionValid(*r.segments[i]));
    EXPECT_LT(distanceBetween(r.segments[i]->end(), r.waypoints[i + 1]), 1e-6);
  }
}

}  // namespace

TEST(RRTStar, ConvergesTowardStraightLineInFreeSpace) {
  auto problem =
      r2Problem(std::make_shared<GeometricWorld2D>(0, 10, 0, 10, 0.0, 0.05));
  RRTStar planner;
  planner.setProblem(problem);
  const PlannerResult r = planner.solve({3000, 0.0, false});
  checkSolution(r, problem);
  const double optimal = std::hypot(8.0, 8.0);
  EXPECT_GE(r.cost, optimal - 1e-9);
  EXPECT_LT(r.cost, optimal * 1.03);
}

TEST(RRTStar, AvoidsObstaclesAndKeepsTreeConsistent) {
  auto problem = r2Problem(wallWorld(0.1));
  RRTStar planner;
  planner.setProblem(problem);
  const PlannerResult r = planner.solve({4000, 0.0, false});
  checkSolution(r, problem);
  checkTreeInvariants(planner, *problem.cost);
}

TEST(RRTStar, CostNeverIncreasesAcrossSolveCalls) {
  RRTStar planner;
  planner.setProblem(r2Problem(wallWorld(0.1)));
  double prev = std::numeric_limits<double>::infinity();
  for (int round = 0; round < 6; ++round) {
    const PlannerResult r = planner.solve({1000, 0.0, false});
    if (r.solved()) {
      EXPECT_LE(r.cost, prev + 1e-9);
      prev = r.cost;
    }
  }
  EXPECT_TRUE(std::isfinite(prev));
}

TEST(RRTStar, StopOnFirstSolution) {
  RRTStar planner;
  planner.setProblem(r2Problem(wallWorld(0.1)));
  const PlannerResult r = planner.solve({100000, 0.0, true});
  ASSERT_TRUE(r.solved());
  EXPECT_EQ(r.iterations, r.first_solution_iteration);
}

TEST(RRTStar, ReportsInvalidStart) {
  auto problem = r2Problem(wallWorld(0.1));
  problem.start = {5.0, 3.0};  // inside the wall
  RRTStar planner;
  planner.setProblem(problem);
  EXPECT_EQ(planner.solve({100, 0.0, false}).status,
            PlannerStatus::kInvalidStart);
}

TEST(RRTStar, RejectsMissingTermination) {
  RRTStar planner;
  planner.setProblem(r2Problem(wallWorld(0.1)));
  EXPECT_EQ(planner.solve({}).status, PlannerStatus::kInvalidProblem);
}

TEST(RRTStar, DubinsPathsRespectHeadingAndCurvature) {
  const double rho = 0.6;
  auto space = std::make_shared<DubinsStateSpace>(rho, 0, 10, 0, 10);
  PlanningProblem problem;
  problem.space = space;
  problem.checker = wallWorld(0.1);
  problem.cost = std::make_shared<PathLengthCost>();
  problem.start = {1, 1, 0};
  problem.goal =
      std::make_shared<GoalState>(space, State{8, 2, kPi / 2}, 0.05);

  RRTStarConfig cfg;
  cfg.max_step = 1.0;
  cfg.max_rewire_radius = 3.0;
  RRTStar planner(cfg);
  planner.setProblem(problem);
  const PlannerResult r = planner.solve({3000, 0.0, false});
  checkSolution(r, problem);
  checkTreeInvariants(planner, *problem.cost);

  // Heading is continuous along the dense path and turns no faster than
  // 1/rho per unit length.
  const double res = 0.01;
  const auto dense = r.densePath(res);
  for (std::size_t i = 1; i < dense.size(); ++i) {
    const double ds = std::hypot(dense[i][0] - dense[i - 1][0],
                                 dense[i][1] - dense[i - 1][1]);
    const double dtheta =
        std::abs(dubins::wrapToPi(dense[i][2] - dense[i - 1][2]));
    EXPECT_LE(dtheta, res / rho + 1e-6);
    EXPECT_LE(ds, res + 1e-6);
  }
}

TEST(RRTStar, CostmapObjectivePlugsIn) {
  // A costly band across the direct route; with a high weight the planner
  // should spend less of its path inside it than when minimising length.
  std::vector<std::string> rows(10, "..........");
  for (int r = 3; r <= 6; ++r) rows[r] = "...999....";
  auto grid = std::make_shared<OccupancyGrid>(OccupancyGrid::fromAscii(rows, 1.0));
  auto checker = std::make_shared<GeometricWorld2D>(0, 10, 0, 10, 0.0, 0.05);

  const auto timeInBand = [&](double weight) {
    auto problem = r2Problem(checker, std::make_shared<CostmapCost>(grid, weight));
    RRTStar planner;
    planner.setProblem(problem);
    const PlannerResult r = planner.solve({3000, 0.0, false});
    EXPECT_TRUE(r.solved());
    double inside = 0.0;
    const auto dense = r.densePath(0.02);
    for (std::size_t i = 1; i < dense.size(); ++i) {
      if (grid->valueAt(dense[i][0], dense[i][1]) > 0.5) {
        inside += std::hypot(dense[i][0] - dense[i - 1][0],
                             dense[i][1] - dense[i - 1][1]);
      }
    }
    return inside;
  };
  EXPECT_LT(timeInBand(20.0), timeInBand(0.0));
}

TEST(RRTStar, ConvergenceHistoryIsMonotone) {
  RRTStarConfig cfg;
  cfg.progress_interval = 100;
  RRTStar planner(cfg);
  planner.setProblem(r2Problem(wallWorld(0.1)));
  planner.solve({1500, 0.0, false});
  const PlannerResult r = planner.solve({1500, 0.0, false});
  ASSERT_TRUE(r.solved());
  ASSERT_GE(r.convergence.size(), 30u);  // periodic samples across both calls
  EXPECT_EQ(r.convergence.back().iteration, 3000u);
  EXPECT_DOUBLE_EQ(r.convergence.back().cost, r.cost);
  EXPECT_LE(r.first_solution_time_s, r.convergence.back().time_s);
  for (std::size_t i = 1; i < r.convergence.size(); ++i) {
    const auto& a = r.convergence[i - 1];
    const auto& b = r.convergence[i];
    EXPECT_GE(b.time_s, a.time_s);
    EXPECT_GT(b.iteration, a.iteration);
    EXPECT_GE(b.tree_size, a.tree_size);
    EXPECT_LE(b.cost, a.cost);
  }
}
