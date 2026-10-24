// Kinodynamic RRT* for a car with a minimum turning radius. Same planner as
// plan_r2.cpp -- only the StateSpace changes.
//
//   ./plan_dubins [iterations] [turning_radius] [output.svg]

#include <cstdio>
#include <cstdlib>
#include <string>

#include "demo_world.hpp"
#include "rrt/planners/rrt_star.hpp"
#include "rrt/spaces/dubins_state_space.hpp"
#include "svg_writer.hpp"

int main(int argc, char** argv) {
  using namespace rrt;
  const std::size_t iterations = argc > 1 ? std::strtoul(argv[1], nullptr, 10)
                                          : 5000;
  const double rho = argc > 2 ? std::strtod(argv[2], nullptr) : 0.6;
  const std::string out = argc > 3 ? argv[3] : "rrt_star_dubins.svg";

  auto space = std::make_shared<DubinsStateSpace>(rho, 0.0, 10.0, 0.0, 10.0);
  auto world = examples::makeDemoWorld(0.15);

  const State start{1.0, 1.0, 0.0};
  // Arrive at the goal heading south, which forces a turn-around manoeuvre.
  const State goal{9.0, 9.0, -1.5707963267948966};

  PlanningProblem problem;
  problem.space = space;
  problem.checker = world;
  problem.cost = std::make_shared<PathLengthCost>();
  problem.start = start;
  problem.goal = std::make_shared<GoalState>(space, goal, 0.05);

  RRTStarConfig config;
  config.max_step = 1.0;
  config.max_rewire_radius = 3.0;
  RRTStar planner(config);
  planner.setProblem(problem);

  Termination term;
  term.max_iterations = iterations;
  const PlannerResult result = planner.solve(term);

  std::printf("%s (Dubins, rho=%.2f): %s after %zu iterations (%.3f s), "
              "%zu vertices\n",
              planner.name().c_str(), rho, toString(result.status).c_str(),
              result.iterations, result.elapsed_s, result.tree_size);
  if (result.solved()) {
    std::printf("  first solution at iteration %zu, final cost %.3f, "
                "%zu waypoints\n",
                result.first_solution_iteration, result.cost,
                result.waypoints.size());
  }

  examples::SvgWriter svg(*world);
  svg.drawWorld();
  svg.drawTree(planner, 0.1);
  svg.drawSolution(result, 0.05);
  svg.drawPose(start, "#2a2");
  svg.drawPose(goal, "#22a");
  if (svg.save(out)) std::printf("  wrote %s\n", out.c_str());
  return result.solved() ? 0 : 1;
}
