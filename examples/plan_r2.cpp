// Geometric RRT* for a point robot in R^2.
//
//   ./plan_r2 [iterations] [output.svg]

#include <cstdio>
#include <cstdlib>
#include <string>

#include "demo_world.hpp"
#include "rrt/planners/rrt_star.hpp"
#include "rrt/spaces/real_vector_2d_space.hpp"
#include "svg_writer.hpp"

int main(int argc, char** argv) {
  using namespace rrt;
  const std::size_t iterations = argc > 1 ? std::strtoul(argv[1], nullptr, 10)
                                          : 5000;
  const std::string out = argc > 2 ? argv[2] : "rrt_star_r2.svg";

  auto space = std::make_shared<RealVector2DSpace>(0.0, 10.0, 0.0, 10.0);
  auto world = examples::makeDemoWorld(0.15);

  PlanningProblem problem;
  problem.space = space;
  problem.checker = world;
  problem.cost = std::make_shared<PathLengthCost>();
  problem.start = {1.0, 1.0};
  problem.goal = std::make_shared<GoalState>(space, State{9.0, 9.0}, 0.1);

  RRTStarConfig config;
  config.max_step = 0.5;
  config.max_rewire_radius = 1.5;
  RRTStar planner(config);
  planner.setProblem(problem);

  Termination term;
  term.max_iterations = iterations;
  const PlannerResult result = planner.solve(term);

  std::printf("%s: %s after %zu iterations (%.3f s), %zu vertices\n",
              planner.name().c_str(), toString(result.status).c_str(),
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
  svg.drawPose(problem.start, "#2a2");
  svg.drawPose({9.0, 9.0}, "#22a");
  if (svg.save(out)) std::printf("  wrote %s\n", out.c_str());
  return result.solved() ? 0 : 1;
}
