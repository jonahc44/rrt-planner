// Terrain cost maps: TerrainMap, TerrainCost, TerrainCollisionChecker and the
// random construction-site generator. Randomised tests draw from a base seed
// that can be changed with RRT_TEST_SEED=<n>; every failure reports the seed
// of the case that failed.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <deque>
#include <memory>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "rrt/cost/terrain.hpp"
#include "rrt/planners/rrt_star.hpp"
#include "rrt/spaces/real_vector_2d_space.hpp"

using namespace rrt;

namespace {

using T = TerrainType;

std::uint64_t baseSeed() {
  static const std::uint64_t seed = [] {
    const char* env = std::getenv("RRT_TEST_SEED");
    return env ? std::strtoull(env, nullptr, 10) : 20261024ULL;
  }();
  return seed;
}

TerrainType terrain(std::size_t i) { return static_cast<TerrainType>(i); }
SiteType site(std::size_t i) { return static_cast<SiteType>(i); }

std::vector<TerrainType> traversableTypes() {
  std::vector<TerrainType> out;
  for (std::size_t i = 0; i < kNumTerrainTypes; ++i)
    if (defaultTerrainProperties(terrain(i)).traversable)
      out.push_back(terrain(i));
  return out;
}

LocalPathPtr line(State a, State b) {
  return std::make_shared<StraightLinePath>(std::move(a), std::move(b));
}

/// Map with every cell drawn uniformly from `types`.
TerrainMap randomCells(std::size_t w, std::size_t h, double res, Rng& rng,
                       const std::vector<TerrainType>& types) {
  TerrainMap m(w, h, res);
  std::uniform_int_distribution<std::size_t> pick(0, types.size() - 1);
  for (std::size_t r = 0; r < h; ++r)
    for (std::size_t c = 0; c < w; ++c) m.set(c, r, types[pick(rng)]);
  return m;
}

/// Exact line integral of the terrain cost along segment a-b: split the
/// segment at every grid line it crosses and weight each piece by its cell.
double exactLineCost(const TerrainMap& m, const State& a, const State& b) {
  const double dx = b[0] - a[0], dy = b[1] - a[1];
  const double len = std::hypot(dx, dy);
  std::vector<double> ts{0.0, 1.0};
  const double res = m.resolution();
  for (std::size_t k = 0; k <= m.width(); ++k) {
    const double gx = m.originX() + static_cast<double>(k) * res;
    if (dx != 0) {
      const double t = (gx - a[0]) / dx;
      if (t > 0 && t < 1) ts.push_back(t);
    }
  }
  for (std::size_t k = 0; k <= m.height(); ++k) {
    const double gy = m.originY() + static_cast<double>(k) * res;
    if (dy != 0) {
      const double t = (gy - a[1]) / dy;
      if (t > 0 && t < 1) ts.push_back(t);
    }
  }
  std::sort(ts.begin(), ts.end());
  double total = 0.0;
  for (std::size_t i = 1; i < ts.size(); ++i) {
    const double tm = 0.5 * (ts[i - 1] + ts[i]);
    total += m.costAt(a[0] + tm * dx, a[1] + tm * dy, 0.0) *
             (ts[i] - ts[i - 1]) * len;
  }
  return total;
}

/// 8-connected flood fill over traversable cells (diagonal moves need both
/// side cells free, so a point robot can actually take them).
bool connected(const TerrainMap& m, const State& a, const State& b) {
  const auto cell = [&](const State& s) {
    return std::pair<long, long>(
        static_cast<long>((s[0] - m.originX()) / m.resolution()),
        static_cast<long>((s[1] - m.originY()) / m.resolution()));
  };
  const long w = static_cast<long>(m.width()), h = static_cast<long>(m.height());
  const auto free = [&](long c, long r) {
    return c >= 0 && r >= 0 && c < w && r < h &&
           m.properties(m.at(static_cast<std::size_t>(c),
                             static_cast<std::size_t>(r))).traversable;
  };
  const auto [sc, sr] = cell(a);
  const auto [gc, gr] = cell(b);
  if (!free(sc, sr) || !free(gc, gr)) return false;
  std::vector<char> seen(static_cast<std::size_t>(w * h), 0);
  std::deque<std::pair<long, long>> q{{sc, sr}};
  seen[static_cast<std::size_t>(sr * w + sc)] = 1;
  while (!q.empty()) {
    const auto [c, r] = q.front();
    q.pop_front();
    if (c == gc && r == gr) return true;
    for (long dr = -1; dr <= 1; ++dr) {
      for (long dc = -1; dc <= 1; ++dc) {
        const long nc = c + dc, nr = r + dr;
        if (!free(nc, nr) || seen[static_cast<std::size_t>(nr * w + nc)])
          continue;
        if (dc && dr && (!free(c + dc, r) || !free(c, r + dr))) continue;
        seen[static_cast<std::size_t>(nr * w + nc)] = 1;
        q.emplace_back(nc, nr);
      }
    }
  }
  return false;
}

std::set<TerrainType> typesPresent(const TerrainMap& m) {
  std::set<TerrainType> out;
  for (std::size_t r = 0; r < m.height(); ++r)
    for (std::size_t c = 0; c < m.width(); ++c) out.insert(m.at(c, r));
  return out;
}

}  // namespace

// ---- Terrain table -----------------------------------------------------------

TEST(TerrainTable, NamesAreUniqueAndCostsAreSane) {
  std::set<std::string> names;
  for (std::size_t i = 0; i < kNumTerrainTypes; ++i) {
    const auto p = defaultTerrainProperties(terrain(i));
    names.insert(terrainName(terrain(i)));
    if (p.traversable) EXPECT_GE(p.cost, 1.0) << terrainName(terrain(i));
  }
  EXPECT_EQ(names.size(), kNumTerrainTypes);
  // Paved surfaces are the reference; soft ground costs more than haul roads.
  EXPECT_DOUBLE_EQ(defaultTerrainProperties(T::kAsphalt).cost, 1.0);
  EXPECT_LT(defaultTerrainProperties(T::kCompactedGravel).cost,
            defaultTerrainProperties(T::kLooseGravel).cost);
  EXPECT_LT(defaultTerrainProperties(T::kPackedDirt).cost,
            defaultTerrainProperties(T::kMud).cost);
  for (const auto t : {T::kExcavation, T::kStockpile, T::kFreshConcrete,
                       T::kStructure})
    EXPECT_FALSE(defaultTerrainProperties(t).traversable) << terrainName(t);
  EXPECT_THROW(terrainName(terrain(kNumTerrainTypes)), std::invalid_argument);
  EXPECT_THROW(siteName(site(kNumSiteTypes)), std::invalid_argument);
}

TEST(TerrainMap, PropertiesCanBeOverriddenPerMap) {
  TerrainMap a(2, 1, 1.0, 0.0, 0.0, T::kMud);
  TerrainMap b = a;
  b.setProperties(T::kMud, {9.0, true});
  EXPECT_DOUBLE_EQ(a.costAt(0.5, 0.5, -1), 5.0);
  EXPECT_DOUBLE_EQ(b.costAt(0.5, 0.5, -1), 9.0);
  b.setProperties(T::kMud, {9.0, false});
  EXPECT_FALSE(b.traversableAt(0.5, 0.5));
  EXPECT_DOUBLE_EQ(b.costAt(0.5, 0.5, -1), -1);  // impassable -> outside
  EXPECT_THROW(b.setProperties(T::kMud, {-1.0, true}), std::invalid_argument);
}

TEST(TerrainMap, LookupRespectsOriginAndBounds) {
  TerrainMap m(4, 2, 0.5, -1.0, 2.0, T::kGrass);
  m.set(3, 1, T::kSand);
  EXPECT_EQ(m.terrainAt(0.9, 2.9), T::kSand);
  EXPECT_EQ(m.terrainAt(-1.0, 2.0), T::kGrass);
  EXPECT_THROW(m.terrainAt(1.0, 2.0), std::out_of_range);
  EXPECT_FALSE(m.traversableAt(-1.01, 2.5));
  EXPECT_DOUBLE_EQ(m.costAt(5, 5, 42.0), 42.0);
  EXPECT_THROW(TerrainMap(0, 1, 1.0), std::invalid_argument);
  EXPECT_THROW(TerrainMap(1, 1, 0.0), std::invalid_argument);
}

TEST(TerrainMap, GridExportsMatchTheTable) {
  Rng rng(baseSeed());
  std::vector<TerrainType> all;
  for (std::size_t i = 0; i < kNumTerrainTypes; ++i) all.push_back(terrain(i));
  const TerrainMap m = randomCells(20, 15, 0.2, rng, all);
  const OccupancyGrid occ = m.occupancyGrid();
  const OccupancyGrid norm = m.normalizedCostGrid();
  double max_cost = 0.0;
  for (const auto t : typesPresent(m))
    if (m.properties(t).traversable)
      max_cost = std::max(max_cost, m.properties(t).cost);
  for (std::size_t r = 0; r < m.height(); ++r) {
    for (std::size_t c = 0; c < m.width(); ++c) {
      const auto& p = m.properties(m.at(c, r));
      EXPECT_EQ(occ.at(c, r), p.traversable ? 0.0 : 1.0);
      EXPECT_DOUBLE_EQ(norm.at(c, r), p.traversable ? p.cost / max_cost : 1.0);
    }
  }
}

// ---- TerrainCost ---------------------------------------------------------------

TEST(TerrainCost, UniformTerrainScalesPathLength) {
  for (const auto t : traversableTypes()) {
    auto m = std::make_shared<TerrainMap>(10, 10, 1.0, 0.0, 0.0, t);
    TerrainCost cost(m);
    const auto p = line({0.5, 0.5}, {9.0, 7.5});
    EXPECT_NEAR(cost.motionCost(*p), p->length() * m->properties(t).cost, 1e-9)
        << terrainName(t);
  }
  auto paved = std::make_shared<TerrainMap>(5, 5, 1.0, 0, 0, T::kAsphalt);
  EXPECT_NEAR(TerrainCost(paved).motionCost(*line({1, 1}, {4, 4})),
              PathLengthCost().motionCost(*line({1, 1}, {4, 4})), 1e-9);
  EXPECT_DOUBLE_EQ(TerrainCost(paved).motionCost(*line({1, 1}, {1, 1})), 0.0);
}

TEST(TerrainCost, ChargesBlockedCostOffTheMapAndOnImpassableCells) {
  auto m = std::make_shared<TerrainMap>(2, 1, 1.0, 0, 0, T::kAsphalt);
  m->set(1, 0, T::kExcavation);
  TerrainCost cost(m, 0.01, /*blocked_cost=*/100.0);
  EXPECT_NEAR(cost.motionCost(*line({0.0, 0.5}, {2.0, 0.5})), 1.0 + 100.0, 1e-9);
  EXPECT_NEAR(cost.motionCost(*line({-1.0, 0.5}, {0.0, 0.5})), 100.0, 1e-9);
}

TEST(TerrainCost, MatchesExactLineIntegralOnRandomTerrain) {
  // Midpoint integration error is bounded by (cell crossings) x step x
  // (largest cost jump); check that bound on random maps and segments.
  const auto types = traversableTypes();
  const double max_cost = 6.0, step = 0.01;
  for (int trial = 0; trial < 50; ++trial) {
    const std::uint64_t seed = baseSeed() + static_cast<std::uint64_t>(trial);
    SCOPED_TRACE("seed " + std::to_string(seed));
    Rng rng(seed);
    std::uniform_real_distribution<double> res_d(0.1, 0.6), u(0.0, 1.0);
    const double res = res_d(rng);
    auto m = std::make_shared<TerrainMap>(
        randomCells(16, 12, res, rng, types));
    const double W = 16 * res, H = 12 * res;
    TerrainCost cost(m, step);
    for (int k = 0; k < 20; ++k) {
      const State a{u(rng) * W, u(rng) * H}, b{u(rng) * W, u(rng) * H};
      const double crossings = 2.0 + 16 + 12;
      EXPECT_NEAR(cost.motionCost(*line(a, b)), exactLineCost(*m, a, b),
                  crossings * step * max_cost);
    }
  }
}

TEST(TerrainCost, IsAdditiveAlongASplitSegment) {
  Rng rng(baseSeed() ^ 0xA5A5);
  auto m = std::make_shared<TerrainMap>(randomCells(30, 30, 0.25, rng,
                                                    traversableTypes()));
  // A step that divides both halves evenly makes the split exact.
  TerrainCost cost(m, 0.001);
  std::uniform_real_distribution<double> u(0.1, 7.4), f(0.1, 0.9);
  for (int k = 0; k < 50; ++k) {
    const State a{u(rng), u(rng)}, b{u(rng), u(rng)};
    const double t = f(rng);
    const State mid{a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])};
    EXPECT_NEAR(cost.motionCost(*line(a, b)),
                cost.motionCost(*line(a, mid)) + cost.motionCost(*line(mid, b)),
                0.1);
  }
}

// ---- TerrainCollisionChecker -------------------------------------------------

TEST(TerrainCollisionChecker, RejectsImpassableTerrainAndLeavingTheMap) {
  auto m = std::make_shared<TerrainMap>(TerrainMap(5, 1, 1.0, 0, 0, T::kGrass));
  m->set(2, 0, T::kStockpile);
  TerrainCollisionChecker c(m, 0.05);
  EXPECT_TRUE(c.isStateValid({1.5, 0.5}));
  EXPECT_FALSE(c.isStateValid({2.5, 0.5}));
  EXPECT_FALSE(c.isStateValid({5.5, 0.5}));
  EXPECT_FALSE(c.isMotionValid(*line({0.5, 0.5}, {4.5, 0.5})));
  EXPECT_TRUE(c.isMotionValid(*line({3.5, 0.5}, {4.5, 0.5})));
  m->set(2, 0, T::kSteelPlate);  // plate the trench over
  EXPECT_TRUE(c.isMotionValid(*line({0.5, 0.5}, {4.5, 0.5})));
}

// ---- Random sites ------------------------------------------------------------

TEST(RandomSite, IsDeterministicInTheSeed) {
  for (std::size_t s = 0; s < kNumSiteTypes; ++s) {
    SCOPED_TRACE(siteName(site(s)));
    const auto a = TerrainMap::randomSite(site(s), 60, 40, 0.25, baseSeed());
    const auto b = TerrainMap::randomSite(site(s), 60, 40, 0.25, baseSeed());
    const auto c = TerrainMap::randomSite(site(s), 60, 40, 0.25, baseSeed() + 1);
    bool same = true, differs = false;
    for (std::size_t r = 0; r < 40; ++r) {
      for (std::size_t col = 0; col < 60; ++col) {
        same &= a.at(col, r) == b.at(col, r);
        differs |= a.at(col, r) != c.at(col, r);
      }
    }
    EXPECT_TRUE(same);
    EXPECT_TRUE(differs);
  }
}

TEST(RandomSite, EachSiteHasItsCharacteristicSurfaces) {
  const std::vector<std::vector<TerrainType>> expected = {
      {T::kAsphalt, T::kExcavation, T::kPackedDirt, T::kGrass},         // road
      {T::kConcreteSlab, T::kStructure, T::kCompactedGravel, T::kMud},  // bldg
      {T::kRubble, T::kStructure, T::kStandingWater, T::kCompactedGravel},
      {T::kCrushedRock, T::kExcavation, T::kLooseGravel, T::kCompactedGravel},
      {T::kExcavation, T::kSteelPlate, T::kStockpile, T::kPackedDirt},  // pipe
  };
  for (std::size_t s = 0; s < kNumSiteTypes; ++s) {
    for (int k = 0; k < 20; ++k) {
      const std::uint64_t seed = baseSeed() + static_cast<std::uint64_t>(k);
      SCOPED_TRACE(std::string(siteName(site(s))) + " seed " +
                   std::to_string(seed));
      const auto m = TerrainMap::randomSite(site(s), 100, 100, 0.1, seed);
      const auto present = typesPresent(m);
      for (const auto t : expected[s])
        EXPECT_TRUE(present.count(t)) << "missing " << terrainName(t);
      // Mixed ground, but not a wall-to-wall obstacle field.
      EXPECT_GE(present.size(), 4u);
      const auto occ = m.occupancyGrid();
      double blocked = 0.0;
      for (std::size_t r = 0; r < 100; ++r)
        for (std::size_t c = 0; c < 100; ++c) blocked += occ.at(c, r);
      EXPECT_LT(blocked / 1e4, 0.35);
    }
  }
}

TEST(RandomSite, KeepClearPointsAreTraversable) {
  std::mt19937_64 rng(baseSeed());
  std::uniform_real_distribution<double> u(0.0, 10.0);
  for (std::size_t s = 0; s < kNumSiteTypes; ++s) {
    for (int k = 0; k < 20; ++k) {
      const std::uint64_t seed = rng();
      SCOPED_TRACE(std::string(siteName(site(s))) + " seed " +
                   std::to_string(seed));
      const std::vector<State> pts{{u(rng), u(rng)}, {u(rng), u(rng)}};
      const auto m = TerrainMap::randomSite(site(s), 50, 50, 0.2, seed, pts,
                                            /*clear_radius=*/0.6, 0.0, 0.0);
      for (const auto& p : pts) {
        for (double dx = -0.5; dx <= 0.5; dx += 0.1) {
          for (double dy = -0.5; dy <= 0.5; dy += 0.1) {
            const double x = p[0] + dx, y = p[1] + dy;
            if (!m.inBounds(x, y) || dx * dx + dy * dy > 0.25) continue;
            EXPECT_TRUE(m.traversableAt(x, y)) << x << ", " << y;
          }
        }
      }
    }
  }
}

TEST(RandomSite, PipelineTrenchAlwaysHasACrossing) {
  // Points on opposite sides of the right-of-way must stay connected.
  for (int k = 0; k < 40; ++k) {
    const std::uint64_t seed = baseSeed() + static_cast<std::uint64_t>(k);
    SCOPED_TRACE("seed " + std::to_string(seed));
    const auto m = TerrainMap::randomSite(SiteType::kPipeline, 80, 80, 0.125,
                                          seed);
    const auto is_free = [&](double x, double y) { return m.traversableAt(x, y); };
    // Corners are far from the right-of-way (it runs through 0.15..0.85).
    const State corners[] = {{0.3, 0.3}, {9.7, 9.7}, {0.3, 9.7}, {9.7, 0.3}};
    std::vector<State> free_corners;
    for (const auto& c : corners)
      if (is_free(c[0], c[1])) free_corners.push_back(c);
    for (std::size_t i = 1; i < free_corners.size(); ++i)
      EXPECT_TRUE(connected(m, free_corners[0], free_corners[i]));
  }
}

// ---- Planning over random sites ------------------------------------------------

TEST(RRTStarTerrain, PrefersCheapGroundOnRandomSites) {
  // On each random site, RRT* minimising terrain cost should find a path
  // whose terrain cost is no worse than the shortest path's (allowing for
  // sampling noise per site), and clearly better on aggregate.
  const State start{0.5, 0.5}, goal{9.5, 9.5};
  double sum_terrain = 0.0, sum_length = 0.0;
  int planned = 0;
  for (std::size_t s = 0; s < kNumSiteTypes; ++s) {
    for (int k = 0; k < 4; ++k) {
      const std::uint64_t seed = baseSeed() * 31 + s * 1000 + static_cast<std::uint64_t>(k);
      SCOPED_TRACE(std::string(siteName(site(s))) + " seed " +
                   std::to_string(seed));
      auto m = std::make_shared<TerrainMap>(TerrainMap::randomSite(
          site(s), 50, 50, 0.2, seed, {start, goal}, 0.5));
      if (!connected(*m, start, goal)) continue;  // generator doesn't promise it

      auto space = std::make_shared<RealVector2DSpace>(0, 10, 0, 10);
      auto checker = std::make_shared<TerrainCollisionChecker>(m, 0.05);
      auto terrain_cost = std::make_shared<TerrainCost>(m, 0.05);
      const auto plan = [&](std::shared_ptr<CostFunction> cost) {
        PlanningProblem p;
        p.space = space;
        p.checker = checker;
        p.cost = std::move(cost);
        p.start = start;
        p.goal = std::make_shared<GoalState>(space, goal, 0.2);
        RRTStarConfig cfg;
        cfg.seed = seed;
        RRTStar planner(cfg);
        planner.setProblem(p);
        return planner.solve({6000, 0.0, false});
      };
      const PlannerResult by_terrain = plan(terrain_cost);
      const PlannerResult by_length = plan(std::make_shared<PathLengthCost>());
      ASSERT_TRUE(by_terrain.solved());
      ASSERT_TRUE(by_length.solved());

      // Every edge passes the checker, and a finer resampling only finds
      // corner clips shorter than the checker's resolution (the usual
      // sampled-motion-check trade-off).
      for (const auto& seg : by_terrain.segments)
        ASSERT_TRUE(checker->isMotionValid(*seg));
      double clipped = 0.0;
      for (const auto& st : by_terrain.densePath(0.005)) {
        clipped = m->traversableAt(st[0], st[1]) ? 0.0 : clipped + 0.005;
        ASSERT_LT(clipped, checker->resolution()) << st[0] << ", " << st[1];
      }

      // The reported cost is the terrain integral along the path.
      double recomputed = 0.0;
      for (const auto& seg : by_terrain.segments)
        recomputed += terrain_cost->motionCost(*seg);
      EXPECT_NEAR(by_terrain.cost, recomputed, 1e-6);

      double length_path_terrain = 0.0;
      for (const auto& seg : by_length.segments)
        length_path_terrain += terrain_cost->motionCost(*seg);
      EXPECT_LE(by_terrain.cost, 1.15 * length_path_terrain);
      sum_terrain += by_terrain.cost;
      sum_length += length_path_terrain;
      ++planned;
    }
  }
  ASSERT_GE(planned, 10) << "too few random sites were connected";
  EXPECT_LT(sum_terrain, sum_length);
}
