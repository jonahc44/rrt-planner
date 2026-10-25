#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "rrt/collision/occupancy_grid.hpp"
#include "rrt/core/cost_function.hpp"
#include "rrt/core/types.hpp"

namespace rrt {

/// Ground surfaces found on construction and work sites. Values are stable
/// (they are stored in TerrainMap cells and exposed to Python).
enum class TerrainType : std::uint8_t {
  kAsphalt = 0,         // finished road / existing paving
  kConcreteSlab,        // cured slab, hardstanding
  kSteelPlate,          // road plates bridging trenches and soft ground
  kCompactedGravel,     // haul roads, laydown yards
  kPackedDirt,          // graded, compacted subgrade
  kGrass,               // undisturbed verge / field
  kLooseGravel,         // fresh aggregate, unrolled base course
  kCrushedRock,         // ballast, riprap
  kSand,                // bedding sand, dune / beach works
  kWetClay,             // rain-softened clay
  kRubble,              // demolition debris
  kMud,                 // churned soft ground
  kStandingWater,       // puddles, dewatering run-off (shallow)
  kExcavation,          // open pit or trench: impassable
  kStockpile,           // spoil / aggregate heap: impassable
  kFreshConcrete,       // uncured pour: impassable
  kStructure,           // building footprint, formwork, scaffolding: impassable
};

constexpr std::size_t kNumTerrainTypes = 17;

/// Per-type traversal parameters.
struct TerrainProperties {
  /// Cost per unit length, relative to asphalt (1.0). Must be >= 0.
  double cost = 1.0;
  /// Whether a vehicle may enter the cell at all.
  bool traversable = true;
};

/// Human-readable name ("asphalt", "wet_clay", ...).
const char* terrainName(TerrainType t);

/// Default table: a wheeled site vehicle (pickup / loader) on each surface.
TerrainProperties defaultTerrainProperties(TerrainType t);

/// Kinds of site the random generator knows how to lay out.
enum class SiteType : std::uint8_t {
  kRoadWorks = 0,   // asphalt corridor, trenches, plated crossings, verges
  kBuilding,        // slab + structure, gravel laydown, mud, stockpiles
  kDemolition,      // rubble field, partial structures, standing water
  kQuarry,          // crushed rock, loose gravel, pits, stockpiles, haul roads
  kPipeline,        // long trench along grass / clay, plated crossings, mud
};

constexpr std::size_t kNumSiteTypes = 5;

const char* siteName(SiteType s);

/// Row-major grid of terrain types plus a cost table. Cell (col, row) covers
/// the same area as in OccupancyGrid (row 0 is the minimum y).
class TerrainMap {
 public:
  TerrainMap(std::size_t width, std::size_t height, double resolution,
             double origin_x = 0.0, double origin_y = 0.0,
             TerrainType fill = TerrainType::kPackedDirt);

  /// Procedurally generated site, deterministic in `seed`. Every point
  /// within `clear_radius` of a point in `keep_clear` is made traversable
  /// (they keep their surface if it already is, else become packed dirt), so
  /// callers can guarantee valid start and goal states. Connectivity between
  /// those points is *not* guaranteed.
  static TerrainMap randomSite(SiteType site, std::size_t width,
                               std::size_t height, double resolution,
                               std::uint64_t seed,
                               const std::vector<State>& keep_clear = {},
                               double clear_radius = 0.5,
                               double origin_x = 0.0, double origin_y = 0.0);

  std::size_t width() const { return width_; }
  std::size_t height() const { return height_; }
  double resolution() const { return resolution_; }
  double originX() const { return origin_x_; }
  double originY() const { return origin_y_; }

  TerrainType at(std::size_t col, std::size_t row) const {
    return cells_[row * width_ + col];
  }
  void set(std::size_t col, std::size_t row, TerrainType t) {
    cells_[row * width_ + col] = t;
  }
  /// Fill the cells whose centres lie in [x0, x1) x [y0, y1).
  void fillRect(double x0, double y0, double x1, double y1, TerrainType t);
  /// Fill the cells whose centres lie within `r` of (cx, cy).
  void fillDisc(double cx, double cy, double r, TerrainType t);

  bool inBounds(double x, double y) const;
  /// Terrain under (x, y). Throws std::out_of_range outside the map.
  TerrainType terrainAt(double x, double y) const;

  const TerrainProperties& properties(TerrainType t) const {
    return table_[static_cast<std::size_t>(t)];
  }
  /// Override the cost / traversability of one terrain type.
  void setProperties(TerrainType t, TerrainProperties p);

  /// Cost per unit length at (x, y); `outside` off the map.
  double costAt(double x, double y, double outside) const;
  /// False for impassable terrain and for points off the map.
  bool traversableAt(double x, double y) const;

  /// 1.0 where impassable, 0.0 elsewhere; feed to GridCollisionChecker.
  OccupancyGrid occupancyGrid() const;
  /// Per-cell costs normalised to [0, 1] by the largest traversable cost
  /// present (impassable cells = 1.0); for use with CostmapCost.
  OccupancyGrid normalizedCostGrid() const;

 private:
  std::size_t width_, height_;
  double resolution_, origin_x_, origin_y_;
  std::vector<TerrainType> cells_;
  std::array<TerrainProperties, kNumTerrainTypes> table_;
};

/// Integral of the terrain cost per unit length along the path, i.e. the
/// path length weighted by the surface under it. On all-asphalt ground (or
/// with every cost = 1) this equals path length. Impassable cells and points
/// off the map are charged `blocked_cost` per unit length; pair this with a
/// collision checker so such motions are rejected outright.
class TerrainCost final : public CostFunction {
 public:
  TerrainCost(std::shared_ptr<const TerrainMap> terrain,
              double integration_step = 0.05, double blocked_cost = 1e3)
      : map_(std::move(terrain)), step_(integration_step),
        blocked_(blocked_cost) {}

  double motionCost(const LocalPath& path) const override;

 private:
  std::shared_ptr<const TerrainMap> map_;
  double step_;
  double blocked_;
};

/// Point robot on a TerrainMap: valid iff the terrain under it is traversable.
class TerrainCollisionChecker final : public CollisionChecker {
 public:
  explicit TerrainCollisionChecker(std::shared_ptr<const TerrainMap> terrain,
                                   double resolution = 0.05)
      : CollisionChecker(resolution), map_(std::move(terrain)) {}

  bool isStateValid(const State& s) const override {
    return map_->traversableAt(s[0], s[1]);
  }

 private:
  std::shared_ptr<const TerrainMap> map_;
};

}  // namespace rrt
