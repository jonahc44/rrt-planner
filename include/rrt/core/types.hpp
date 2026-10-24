#pragma once

#include <random>
#include <vector>

namespace rrt {

/// A point in a state space. Layout is defined by the StateSpace that owns it
/// (e.g. [x, y] for R^2, [x, y, theta] for SE(2)). By convention, spaces that
/// live in a planar workspace put x and y in components 0 and 1, which is what
/// the 2D collision checkers and cost maps read.
using State = std::vector<double>;

using Rng = std::mt19937_64;

}  // namespace rrt
