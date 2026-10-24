#include <gtest/gtest.h>

#include <algorithm>

#include "rrt/core/nearest_neighbors.hpp"
#include "rrt/spaces/dubins_state_space.hpp"
#include "rrt/spaces/real_vector_2d_space.hpp"

using namespace rrt;

namespace {

std::vector<std::size_t> sorted(std::vector<std::size_t> v) {
  std::sort(v.begin(), v.end());
  return v;
}

/// The KD-tree must agree exactly with brute force, for symmetric and
/// asymmetric metrics alike.
void checkAgainstBruteForce(const std::shared_ptr<const StateSpace>& space,
                            double radius) {
  KdTreeNearestNeighbors kd(space);
  BruteForceNearestNeighbors bf(space);
  Rng rng(7);
  std::vector<State> states;
  for (int i = 0; i < 600; ++i) {
    const State s = space->sampleUniform(rng);
    states.push_back(s);
    ASSERT_EQ(kd.add(s), bf.add(s));
    if (i % 20 != 0) continue;
    for (int q = 0; q < 10; ++q) {
      const State query = space->sampleUniform(rng);
      // Compare distances, not ids, while the tree is still growing.
      EXPECT_DOUBLE_EQ(space->distance(states[kd.nearest(query)], query),
                       space->distance(states[bf.nearest(query)], query));
    }
  }
  for (int q = 0; q < 200; ++q) {
    const State query = space->sampleUniform(rng);
    EXPECT_EQ(kd.nearest(query), bf.nearest(query));
    EXPECT_EQ(sorted(kd.nearTo(query, radius)),
              sorted(bf.nearTo(query, radius)));
    EXPECT_EQ(sorted(kd.nearFrom(query, radius)),
              sorted(bf.nearFrom(query, radius)));
  }
}

}  // namespace

TEST(NearestNeighbors, KdTreeMatchesBruteForceInR2) {
  checkAgainstBruteForce(
      std::make_shared<RealVector2DSpace>(0, 10, 0, 10), 1.0);
}

TEST(NearestNeighbors, KdTreeMatchesBruteForceForDubins) {
  checkAgainstBruteForce(
      std::make_shared<DubinsStateSpace>(1.0, 0, 10, 0, 10), 3.0);
}

TEST(NearestNeighbors, ClearResets) {
  auto space = std::make_shared<RealVector2DSpace>(0, 1, 0, 1);
  KdTreeNearestNeighbors kd(space);
  kd.add({0.5, 0.5});
  kd.clear();
  EXPECT_EQ(kd.size(), 0u);
  EXPECT_EQ(kd.add({0.1, 0.1}), 0u);
  EXPECT_EQ(kd.nearest({0.0, 0.0}), 0u);
}
