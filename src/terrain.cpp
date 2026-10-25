#include "rrt/cost/terrain.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

namespace rrt {

namespace {

using T = TerrainType;

struct TerrainInfo {
  const char* name;
  TerrainProperties props;
};

// Costs are relative effort per metre for a wheeled site vehicle: rolling
// resistance, the speed it is safe to hold, and wear / risk of getting stuck.
constexpr std::array<TerrainInfo, kNumTerrainTypes> kTerrainInfo = {{
    {"asphalt", {1.0, true}},
    {"concrete_slab", {1.0, true}},
    {"steel_plate", {1.3, true}},
    {"compacted_gravel", {1.2, true}},
    {"packed_dirt", {1.5, true}},
    {"grass", {1.8, true}},
    {"loose_gravel", {2.2, true}},
    {"crushed_rock", {2.8, true}},
    {"sand", {3.0, true}},
    {"wet_clay", {4.0, true}},
    {"rubble", {4.5, true}},
    {"mud", {5.0, true}},
    {"standing_water", {6.0, true}},
    {"excavation", {0.0, false}},
    {"stockpile", {0.0, false}},
    {"fresh_concrete", {0.0, false}},
    {"structure", {0.0, false}},
}};

constexpr const char* kSiteNames[kNumSiteTypes] = {
    "road_works", "building", "demolition", "quarry", "pipeline"};

std::size_t index(TerrainType t) {
  const auto i = static_cast<std::size_t>(t);
  if (i >= kNumTerrainTypes) throw std::invalid_argument("bad TerrainType");
  return i;
}

/// Painting primitives in world coordinates, plus the RNG that places them.
class SitePainter {
 public:
  SitePainter(TerrainMap& map, std::uint64_t seed)
      : m_(map), rng_(seed),
        w_(static_cast<double>(map.width()) * map.resolution()),
        h_(static_cast<double>(map.height()) * map.resolution()),
        x0_(map.originX()), y0_(map.originY()) {}

  double W() const { return w_; }
  double H() const { return h_; }
  double S() const { return std::min(w_, h_); }  // feature scale

  // Hand-rolled draws (not <random> distributions, whose algorithms differ
  // between standard libraries) so a seed gives the same site everywhere.
  // Never make two draws in one call's argument list: evaluation order is
  // unspecified and differs between compilers.
  double uni(double lo, double hi) {
    return lo + (hi - lo) * static_cast<double>(rng_() >> 11) * 0x1p-53;
  }
  int count(int lo, int hi) {
    return lo + static_cast<int>(rng_() % static_cast<std::uint64_t>(hi - lo + 1));
  }
  bool coin(double p = 0.5) { return uni(0.0, 1.0) < p; }
  /// Random point, as a fraction of the map, in world coordinates.
  double rx(double lo = 0.0, double hi = 1.0) { return wx(uni(lo, hi)); }
  double ry(double lo = 0.0, double hi = 1.0) { return wy(uni(lo, hi)); }
  /// Fraction of the map -> world coordinate.
  double wx(double f) const { return x0_ + w_ * f; }
  double wy(double f) const { return y0_ + h_ * f; }

  // Coordinates below are relative to the map origin, as fractions of W / H.
  void rect(double fx0, double fy0, double fx1, double fy1, TerrainType t) {
    m_.fillRect(x0_ + fx0 * w_, y0_ + fy0 * h_, x0_ + fx1 * w_,
                y0_ + fy1 * h_, t);
  }

  /// Cells whose centre is within `r` (world units) of segment a-b.
  void capsule(double ax, double ay, double bx, double by, double r,
               TerrainType t) {
    const double dx = bx - ax, dy = by - ay;
    const double len2 = dx * dx + dy * dy;
    forCellsIn(std::min(ax, bx) - r, std::min(ay, by) - r,
               std::max(ax, bx) + r, std::max(ay, by) + r,
               [&](std::size_t c, std::size_t row, double x, double y) {
                 double u = len2 > 0 ? ((x - ax) * dx + (y - ay) * dy) / len2
                                     : 0.0;
                 u = std::clamp(u, 0.0, 1.0);
                 const double ex = x - (ax + u * dx), ey = y - (ay + u * dy);
                 if (ex * ex + ey * ey <= r * r) m_.set(c, row, t);
               });
  }

  /// Irregular patch: a cluster of overlapping discs around (cx, cy). Discs
  /// are at least 0.75 cells across in radius so no feature vanishes on a
  /// coarse grid.
  void blob(double cx, double cy, double size, TerrainType t) {
    const double min_r = 0.75 * m_.resolution();
    const int n = count(3, 6);
    for (int i = 0; i < n; ++i) {
      const double a = uni(0.0, 6.283185307179586);
      const double d = uni(0.0, 0.6 * size);
      const double r = std::max(min_r, uni(0.35, 0.7) * size);
      m_.fillDisc(cx + d * std::cos(a), cy + d * std::sin(a), r, t);
    }
  }

  /// `lo..hi` blobs of `a` (or `b` with probability `p_b`) at random spots
  /// in the [f_lo, f_hi]^2 fraction of the map; size is a fraction of S().
  void scatter(int lo, int hi, double size_lo, double size_hi, TerrainType a,
               TerrainType b, double p_b, double f_lo = 0.0,
               double f_hi = 1.0) {
    for (int i = count(lo, hi); i > 0; --i) {
      const double x = rx(f_lo, f_hi);
      const double y = ry(f_lo, f_hi);
      const double size = uni(size_lo, size_hi) * S();
      blob(x, y, size, coin(p_b) ? b : a);
    }
  }
  void scatter(int lo, int hi, double size_lo, double size_hi, TerrainType t,
               double f_lo = 0.0, double f_hi = 1.0) {
    scatter(lo, hi, size_lo, size_hi, t, t, 0.0, f_lo, f_hi);
  }

  /// Polyline with random kinks between two points (haul roads, trenches).
  void track(double ax, double ay, double bx, double by, double r,
             TerrainType t, int kinks) {
    double px = ax, py = ay;
    for (int i = 1; i <= kinks + 1; ++i) {
      const double f = static_cast<double>(i) / (kinks + 1);
      double nx = ax + f * (bx - ax), ny = ay + f * (by - ay);
      if (i <= kinks) {
        nx += uni(-0.12, 0.12) * w_;
        ny += uni(-0.12, 0.12) * h_;
      }
      capsule(px, py, nx, ny, r, t);
      px = nx;
      py = ny;
    }
  }

  template <typename F>
  void forCellsIn(double xa, double ya, double xb, double yb, F&& f) {
    const double res = m_.resolution();
    const auto lo = [&](double v, double o) {
      return static_cast<long>(std::floor((v - o) / res - 0.5));
    };
    const auto hi = [&](double v, double o) {
      return static_cast<long>(std::ceil((v - o) / res - 0.5));
    };
    const long c0 = std::max(0L, lo(xa, x0_));
    const long r0 = std::max(0L, lo(ya, y0_));
    const long c1 = std::min(static_cast<long>(m_.width()) - 1, hi(xb, x0_));
    const long r1 = std::min(static_cast<long>(m_.height()) - 1, hi(yb, y0_));
    for (long r = r0; r <= r1; ++r) {
      for (long c = c0; c <= c1; ++c) {
        f(static_cast<std::size_t>(c), static_cast<std::size_t>(r),
          x0_ + (static_cast<double>(c) + 0.5) * res,
          y0_ + (static_cast<double>(r) + 0.5) * res);
      }
    }
  }

  TerrainMap& map() { return m_; }

 private:
  TerrainMap& m_;
  Rng rng_;
  double w_, h_, x0_, y0_;
};

void paintRoadWorks(SitePainter& p) {
  // Asphalt carriageway across the site, packed-dirt shoulders, grass verges.
  const bool horizontal = p.coin();
  const double c = p.uni(0.35, 0.65), half = p.uni(0.10, 0.16);
  const double sh = half + p.uni(0.04, 0.08);
  if (horizontal) {
    p.rect(0, c - sh, 1, c + sh, T::kPackedDirt);
    p.rect(0, c - half, 1, c + half, T::kAsphalt);
  } else {
    p.rect(c - sh, 0, c + sh, 1, T::kPackedDirt);
    p.rect(c - half, 0, c + half, 1, T::kAsphalt);
  }
  // Milled-off sections awaiting resurfacing.
  for (int i = p.count(1, 3); i > 0; --i) {
    const double a = p.uni(0.05, 0.8), len = p.uni(0.08, 0.2);
    if (horizontal) p.rect(a, c - half, a + len, c, T::kLooseGravel);
    else p.rect(c, a, c + half, a + len, T::kLooseGravel);
  }
  // Utility trenches cut across the road; some are plated over.
  for (int i = p.count(1, 3); i > 0; --i) {
    const double a = p.uni(0.1, 0.9), tw = p.uni(0.015, 0.03);
    const double ext = sh + p.uni(0.0, 0.15);
    if (horizontal) p.rect(a - tw, c - ext, a + tw, c + ext, T::kExcavation);
    else p.rect(c - ext, a - tw, c + ext, a + tw, T::kExcavation);
    if (p.coin(0.7)) {
      const double pw = tw + 0.02, ph = p.uni(0.4, 1.0) * half;
      if (horizontal) p.rect(a - pw, c - ph, a + pw, c + ph, T::kSteelPlate);
      else p.rect(c - ph, a - pw, c + ph, a + pw, T::kSteelPlate);
    }
  }
  // Soft verges and a spoil heap or two.
  p.scatter(2, 5, 0.04, 0.1, T::kMud, T::kWetClay, 0.5);
  p.scatter(0, 2, 0.03, 0.06, T::kStockpile);
}

void paintBuilding(SitePainter& p) {
  const double s = p.S();
  // Slab with the rising structure on part of it and a pour in progress.
  const double bx = p.uni(0.25, 0.45), by = p.uni(0.25, 0.45);
  const double bw = p.uni(0.25, 0.35), bh = p.uni(0.25, 0.35);
  p.rect(bx - 0.04, by - 0.04, bx + bw + 0.04, by + bh + 0.04, T::kConcreteSlab);
  const double built = p.uni(0.4, 0.8);
  p.rect(bx, by, bx + bw * built, by + bh, T::kStructure);
  if (p.coin(0.6)) {
    const double poured = p.uni(0.3, 0.9);
    p.rect(bx + bw * 0.82, by, bx + bw, by + bh * poured, T::kFreshConcrete);
  }
  // Gravel haul road from a site entrance, and a laydown yard.
  {
    const double ax = p.rx(0.1, 0.9), ay = p.ry(-0.05, 0.0);
    const double bx2 = p.rx(0.2, 0.8), by2 = p.ry(0.6, 1.05);
    const double r = p.uni(0.03, 0.05) * s;
    p.track(ax, ay, bx2, by2, r, T::kCompactedGravel, p.count(1, 2));
  }
  const double lx = p.coin() ? p.uni(0.0, 0.1) : p.uni(0.7, 0.8);
  const double ly = p.uni(0.55, 0.8);
  const double lw = p.uni(0.12, 0.2), lh = p.uni(0.1, 0.2);
  p.rect(lx, ly, lx + lw, ly + lh, T::kCompactedGravel);
  // Footing excavations, ruts, puddles, heaps.
  p.scatter(0, 2, 0.03, 0.05, T::kExcavation, 0.1, 0.9);
  p.scatter(2, 5, 0.04, 0.1, T::kMud);
  p.scatter(0, 3, 0.02, 0.04, T::kStandingWater);
  p.scatter(1, 3, 0.03, 0.06, T::kStockpile, T::kSand, 0.5);
}

void paintDemolition(SitePainter& p) {
  const double s = p.S();
  // Rubble field over the old footprint, with slab remnants.
  p.scatter(1, 1, 0.3, 0.45, T::kRubble, 0.35, 0.65);
  for (int i = p.count(1, 3); i > 0; --i) {
    const double x = p.uni(0.1, 0.7), y = p.uni(0.1, 0.7);
    const double w = p.uni(0.1, 0.2), h = p.uni(0.1, 0.2);
    p.rect(x, y, x + w, y + h, T::kConcreteSlab);
  }
  // Gravel access road for the trucks; debris, puddles and walls still
  // standing end up on it too, so it goes down first.
  {
    const double ax = p.rx(-0.05, 0.0), ay = p.ry(0.1, 0.9);
    const double bx = p.rx(1.0, 1.05), by = p.ry(0.1, 0.9);
    const double r = p.uni(0.03, 0.05) * s;
    p.track(ax, ay, bx, by, r, T::kCompactedGravel, p.count(0, 2));
  }
  // Crushed-concrete piles, sorted stockpiles, flooded basement corners.
  p.scatter(1, 3, 0.04, 0.08, T::kCrushedRock);
  p.scatter(1, 2, 0.03, 0.06, T::kStockpile);
  p.scatter(1, 3, 0.03, 0.07, T::kStandingWater);
  // Walls not yet pulled down; thin, so painted last to stay intact.
  for (int i = p.count(1, 4); i > 0; --i) {
    const double ax = p.rx(0.15, 0.85), ay = p.ry(0.15, 0.85);
    const double len = p.uni(0.1, 0.25) * s;
    const bool along_x = p.coin();
    const double r = p.uni(0.01, 0.02) * s;
    p.capsule(ax, ay, ax + (along_x ? len : 0), ay + (along_x ? 0 : len), r,
              T::kStructure);
  }
}

void paintQuarry(SitePainter& p) {
  const double s = p.S();
  // Working benches of crushed rock around one or two pits.
  p.scatter(1, 1, 0.35, 0.5, T::kCrushedRock, 0.3, 0.7);
  for (int i = p.count(1, 2); i > 0; --i) {
    const double cx = p.rx(0.2, 0.8), cy = p.ry(0.2, 0.8);
    const double r = p.uni(0.08, 0.14) * s;
    p.map().fillDisc(cx, cy, r * 1.5, T::kLooseGravel);
    p.blob(cx, cy, r, T::kExcavation);
  }
  // Product stockpiles.
  p.scatter(2, 4, 0.03, 0.07, T::kStockpile, T::kSand, 0.3);
  // Pit-floor sump and run-off.
  p.scatter(0, 2, 0.03, 0.06, T::kStandingWater);
  p.scatter(1, 3, 0.03, 0.08, T::kMud);
  // Haul roads, painted last so they stay continuous.
  for (int i = p.count(1, 2); i > 0; --i) {
    const double ax = p.rx(0.0, 0.2), ay = p.ry(0.0, 1.0);
    const double bx = p.rx(0.8, 1.0), by = p.ry(0.0, 1.0);
    const double r = p.uni(0.03, 0.05) * s;
    p.track(ax, ay, bx, by, r, T::kCompactedGravel, p.count(1, 3));
  }
}

void paintPipeline(SitePainter& p) {
  const double s = p.S();
  p.scatter(2, 4, 0.08, 0.15, T::kWetClay);
  // Right-of-way running across the site: working strip, the trench, and a
  // spoil row beside it.
  const bool horizontal = p.coin();
  const double a = p.uni(0.3, 0.7), b = a + p.uni(-0.15, 0.15);
  // World point at fraction f along the route, `off` (fraction) across it.
  const auto pt = [&](double f, double off) {
    const double c = a + f * (b - a) + off;
    return horizontal ? std::pair{p.wx(f), p.wy(c)}
                      : std::pair{p.wx(c), p.wy(f)};
  };
  const auto seg = [&](double f0, double off0, double f1, double off1,
                       double r, TerrainType t) {
    const auto [ax, ay] = pt(f0, off0);
    const auto [bx, by] = pt(f1, off1);
    p.capsule(ax, ay, bx, by, r, t);
  };
  seg(-0.05, 0.0, 1.05, 0.0, 0.10 * s, T::kPackedDirt);
  // Ruts along the working strip.
  for (int i = p.count(2, 4); i > 0; --i) {
    const double f = p.uni(0.0, 1.0), off = p.uni(-0.08, 0.08);
    const auto [x, y] = pt(f, off);
    p.blob(x, y, p.uni(0.03, 0.06) * s, T::kMud);
  }
  // Access track from the site edge to the right-of-way.
  const double fa = p.uni(0.2, 0.8);
  const auto [tx, ty] = pt(fa, 0.0);
  const double edge = p.coin() ? -0.05 : 1.05;
  p.track(tx, ty, horizontal ? tx : p.wx(edge), horizontal ? p.wy(edge) : ty,
          0.03 * s, T::kCompactedGravel, 1);
  seg(-0.05, 0.0, 1.05, 0.0, 0.02 * s, T::kExcavation);
  // Spoil row on one side, broken at intervals.
  const double side = p.coin() ? 0.06 : -0.06;
  for (double f = 0.0; f < 1.0; f += 0.25) {
    const double g = f + p.uni(0.03, 0.08);
    const double len = p.uni(0.08, 0.15);
    seg(g, side, g + len, side, 0.015 * s, T::kStockpile);
  }
  // Plated crossings: always one where the access track arrives, so the
  // trench never seals off part of the site.
  seg(fa, -0.09, fa, 0.09, 0.04 * s, T::kSteelPlate);
  for (int i = p.count(0, 2); i > 0; --i) {
    const double f = p.uni(0.1, 0.9);
    seg(f, -0.09, f, 0.09, 0.04 * s, T::kSteelPlate);
  }
}

}  // namespace

const char* terrainName(TerrainType t) { return kTerrainInfo[index(t)].name; }

TerrainProperties defaultTerrainProperties(TerrainType t) {
  return kTerrainInfo[index(t)].props;
}

const char* siteName(SiteType s) {
  const auto i = static_cast<std::size_t>(s);
  if (i >= kNumSiteTypes) throw std::invalid_argument("bad SiteType");
  return kSiteNames[i];
}

TerrainMap::TerrainMap(std::size_t width, std::size_t height,
                       double resolution, double origin_x, double origin_y,
                       TerrainType fill)
    : width_(width), height_(height), resolution_(resolution),
      origin_x_(origin_x), origin_y_(origin_y), cells_(width * height, fill) {
  if (width == 0 || height == 0 || !(resolution > 0.0))
    throw std::invalid_argument("TerrainMap: empty grid or bad resolution");
  index(fill);
  for (std::size_t i = 0; i < kNumTerrainTypes; ++i)
    table_[i] = kTerrainInfo[i].props;
}

void TerrainMap::fillRect(double x0, double y0, double x1, double y1,
                          TerrainType t) {
  index(t);
  for (std::size_t r = 0; r < height_; ++r) {
    const double y = origin_y_ + (static_cast<double>(r) + 0.5) * resolution_;
    if (y < y0 || y >= y1) continue;
    for (std::size_t c = 0; c < width_; ++c) {
      const double x = origin_x_ + (static_cast<double>(c) + 0.5) * resolution_;
      if (x >= x0 && x < x1) set(c, r, t);
    }
  }
}

void TerrainMap::fillDisc(double cx, double cy, double rad, TerrainType t) {
  index(t);
  for (std::size_t r = 0; r < height_; ++r) {
    const double dy =
        origin_y_ + (static_cast<double>(r) + 0.5) * resolution_ - cy;
    if (std::abs(dy) > rad) continue;
    for (std::size_t c = 0; c < width_; ++c) {
      const double dx =
          origin_x_ + (static_cast<double>(c) + 0.5) * resolution_ - cx;
      if (dx * dx + dy * dy <= rad * rad) set(c, r, t);
    }
  }
}

bool TerrainMap::inBounds(double x, double y) const {
  return x >= origin_x_ && y >= origin_y_ &&
         x < origin_x_ + static_cast<double>(width_) * resolution_ &&
         y < origin_y_ + static_cast<double>(height_) * resolution_;
}

TerrainType TerrainMap::terrainAt(double x, double y) const {
  if (!inBounds(x, y)) throw std::out_of_range("terrainAt: outside the map");
  const auto col = std::min(
      width_ - 1, static_cast<std::size_t>((x - origin_x_) / resolution_));
  const auto row = std::min(
      height_ - 1, static_cast<std::size_t>((y - origin_y_) / resolution_));
  return at(col, row);
}

void TerrainMap::setProperties(TerrainType t, TerrainProperties p) {
  if (!(p.cost >= 0.0))
    throw std::invalid_argument("terrain cost must be non-negative");
  table_[index(t)] = p;
}

double TerrainMap::costAt(double x, double y, double outside) const {
  if (!inBounds(x, y)) return outside;
  const auto& p = properties(terrainAt(x, y));
  return p.traversable ? p.cost : outside;
}

bool TerrainMap::traversableAt(double x, double y) const {
  return inBounds(x, y) && properties(terrainAt(x, y)).traversable;
}

OccupancyGrid TerrainMap::occupancyGrid() const {
  OccupancyGrid g(width_, height_, resolution_, origin_x_, origin_y_);
  for (std::size_t r = 0; r < height_; ++r)
    for (std::size_t c = 0; c < width_; ++c)
      g.at(c, r) = properties(at(c, r)).traversable ? 0.0 : 1.0;
  return g;
}

OccupancyGrid TerrainMap::normalizedCostGrid() const {
  double max_cost = 0.0;
  for (const auto t : cells_) {
    const auto& p = properties(t);
    if (p.traversable) max_cost = std::max(max_cost, p.cost);
  }
  OccupancyGrid g(width_, height_, resolution_, origin_x_, origin_y_);
  for (std::size_t r = 0; r < height_; ++r) {
    for (std::size_t c = 0; c < width_; ++c) {
      const auto& p = properties(at(c, r));
      g.at(c, r) = !p.traversable ? 1.0
                   : max_cost > 0 ? p.cost / max_cost
                                  : 0.0;
    }
  }
  return g;
}

TerrainMap TerrainMap::randomSite(SiteType site, std::size_t width,
                                  std::size_t height, double resolution,
                                  std::uint64_t seed,
                                  const std::vector<State>& keep_clear,
                                  double clear_radius, double origin_x,
                                  double origin_y) {
  const TerrainType base = [&] {
    switch (site) {
      case SiteType::kRoadWorks: return T::kGrass;
      case SiteType::kBuilding: return T::kPackedDirt;
      case SiteType::kDemolition: return T::kPackedDirt;
      case SiteType::kQuarry: return T::kPackedDirt;
      case SiteType::kPipeline: return T::kGrass;
    }
    throw std::invalid_argument("bad SiteType");
  }();
  TerrainMap map(width, height, resolution, origin_x, origin_y, base);
  // Mix the site type into the seed so seed k differs across site types.
  SitePainter p(map, seed * 0x9E3779B97F4A7C15ULL + static_cast<int>(site));
  switch (site) {
    case SiteType::kRoadWorks: paintRoadWorks(p); break;
    case SiteType::kBuilding: paintBuilding(p); break;
    case SiteType::kDemolition: paintDemolition(p); break;
    case SiteType::kQuarry: paintQuarry(p); break;
    case SiteType::kPipeline: paintPipeline(p); break;
  }
  for (const auto& s : keep_clear) {
    if (s.size() < 2) throw std::invalid_argument("keep_clear: need x, y");
    p.forCellsIn(s[0] - clear_radius, s[1] - clear_radius,
                 s[0] + clear_radius, s[1] + clear_radius,
                 [&](std::size_t c, std::size_t r, double x, double y) {
                   // Distance from the point to the nearest part of the
                   // cell, so every point within the radius is covered.
                   const double h = 0.5 * resolution;
                   const double dx = std::max(0.0, std::abs(x - s[0]) - h);
                   const double dy = std::max(0.0, std::abs(y - s[1]) - h);
                   if (dx * dx + dy * dy <= clear_radius * clear_radius &&
                       !map.properties(map.at(c, r)).traversable)
                     map.set(c, r, T::kPackedDirt);
                 });
  }
  return map;
}

double TerrainCost::motionCost(const LocalPath& path) const {
  const double len = path.length();
  if (len <= 0.0) return 0.0;
  const auto n =
      static_cast<std::size_t>(std::max(1.0, std::ceil(len / step_)));
  const double ds = len / static_cast<double>(n);
  // Midpoint rule over n sub-intervals.
  double total = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const State s = path.interpolate((static_cast<double>(i) + 0.5) * ds);
    total += map_->costAt(s[0], s[1], blocked_);
  }
  return total * ds;
}

}  // namespace rrt
