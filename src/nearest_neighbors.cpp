#include "rrt/core/nearest_neighbors.hpp"

#include <limits>
#include <stdexcept>

#include <nanoflann.hpp>

namespace rrt {

// ---------------------------------------------------------------------------
// Brute force
// ---------------------------------------------------------------------------

std::size_t BruteForceNearestNeighbors::add(const State& s) {
  states_.push_back(s);
  return states_.size() - 1;
}

std::size_t BruteForceNearestNeighbors::nearest(const State& query) const {
  if (states_.empty()) throw std::logic_error("nearest() on empty index");
  std::size_t best = 0;
  double best_d = std::numeric_limits<double>::infinity();
  for (std::size_t i = 0; i < states_.size(); ++i) {
    const double d = space_->distance(states_[i], query);
    if (d < best_d) {
      best_d = d;
      best = i;
    }
  }
  return best;
}

std::vector<std::size_t> BruteForceNearestNeighbors::nearTo(const State& query,
                                                            double r) const {
  std::vector<std::size_t> out;
  for (std::size_t i = 0; i < states_.size(); ++i) {
    if (space_->distance(states_[i], query) <= r) out.push_back(i);
  }
  return out;
}

std::vector<std::size_t> BruteForceNearestNeighbors::nearFrom(
    const State& query, double r) const {
  std::vector<std::size_t> out;
  for (std::size_t i = 0; i < states_.size(); ++i) {
    if (space_->distance(query, states_[i]) <= r) out.push_back(i);
  }
  return out;
}

// ---------------------------------------------------------------------------
// nanoflann KD-tree
// ---------------------------------------------------------------------------

namespace {

/// nanoflann dataset adaptor over a flat array of embedded points.
struct EmbeddedCloud {
  std::size_t dim = 0;
  std::vector<double> coords;  // size() * dim
  std::vector<State> states;   // original states, for exact filtering

  std::size_t kdtree_get_point_count() const { return states.size(); }
  double kdtree_get_pt(std::size_t idx, std::size_t d) const {
    return coords[idx * dim + d];
  }
  template <class BBox>
  bool kdtree_get_bbox(BBox&) const {
    return false;
  }
};

using KdTree = nanoflann::KDTreeSingleIndexDynamicAdaptor<
    nanoflann::L2_Simple_Adaptor<double, EmbeddedCloud, double, std::size_t>,
    EmbeddedCloud, -1, std::size_t>;

// Inflate embedded search radii slightly so floating-point error in the
// embedding never drops a true neighbour (results are filtered exactly
// afterwards, so this only costs a few extra distance evaluations).
constexpr double kRadiusSlack = 1e-9;

}  // namespace

struct KdTreeNearestNeighbors::Impl {
  EmbeddedCloud cloud;
  std::unique_ptr<KdTree> tree;
  std::vector<double> scratch;  // embedding of the current query

  void reset() {
    cloud.coords.clear();
    cloud.states.clear();
    tree = std::make_unique<KdTree>(
        static_cast<int>(cloud.dim), cloud,
        nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */));
  }
};

KdTreeNearestNeighbors::KdTreeNearestNeighbors(
    std::shared_ptr<const StateSpace> space)
    : space_(std::move(space)), impl_(std::make_unique<Impl>()) {
  impl_->cloud.dim = space_->embeddingDimension();
  impl_->scratch.resize(impl_->cloud.dim);
  impl_->reset();
}

KdTreeNearestNeighbors::~KdTreeNearestNeighbors() = default;

std::size_t KdTreeNearestNeighbors::add(const State& s) {
  auto& cloud = impl_->cloud;
  const std::size_t id = cloud.states.size();
  cloud.coords.resize(cloud.coords.size() + cloud.dim);
  space_->embed(s, &cloud.coords[id * cloud.dim]);
  cloud.states.push_back(s);
  impl_->tree->addPoints(id, id);
  return id;
}

std::size_t KdTreeNearestNeighbors::size() const {
  return impl_->cloud.states.size();
}

void KdTreeNearestNeighbors::clear() { impl_->reset(); }

std::vector<std::size_t> KdTreeNearestNeighbors::embeddedRadius(
    const State& query, double r) const {
  space_->embed(query, impl_->scratch.data());
  const double r_sq = r * r * (1.0 + kRadiusSlack) + kRadiusSlack;
  std::vector<nanoflann::ResultItem<std::size_t, double>> hits;
  nanoflann::RadiusResultSet<double, std::size_t> result(r_sq, hits);
  impl_->tree->findNeighbors(result, impl_->scratch.data());
  std::vector<std::size_t> ids;
  ids.reserve(hits.size());
  for (const auto& h : hits) ids.push_back(h.first);
  return ids;
}

std::size_t KdTreeNearestNeighbors::nearest(const State& query) const {
  const auto& states = impl_->cloud.states;
  if (states.empty()) throw std::logic_error("nearest() on empty index");

  // 1. Euclidean nearest neighbour in the embedding: a candidate.
  space_->embed(query, impl_->scratch.data());
  std::size_t cand = 0;
  double cand_sq = 0.0;
  nanoflann::KNNResultSet<double, std::size_t> knn(1);
  knn.init(&cand, &cand_sq);
  impl_->tree->findNeighbors(knn, impl_->scratch.data());

  // 2. Any state closer than the candidate under the true metric must lie
  //    within that distance in the embedding (lower-bound property), so a
  //    radius search around the query finds the exact answer.
  const double cand_d = space_->distance(states[cand], query);
  std::size_t best = cand;
  double best_d = cand_d;
  for (std::size_t id : embeddedRadius(query, cand_d)) {
    if (id == cand) continue;
    const double d = space_->distance(states[id], query);
    if (d < best_d) {
      best_d = d;
      best = id;
    }
  }
  return best;
}

std::vector<std::size_t> KdTreeNearestNeighbors::nearTo(const State& query,
                                                        double r) const {
  const auto& states = impl_->cloud.states;
  std::vector<std::size_t> out;
  for (std::size_t id : embeddedRadius(query, r)) {
    if (space_->distance(states[id], query) <= r) out.push_back(id);
  }
  return out;
}

std::vector<std::size_t> KdTreeNearestNeighbors::nearFrom(const State& query,
                                                          double r) const {
  const auto& states = impl_->cloud.states;
  std::vector<std::size_t> out;
  for (std::size_t id : embeddedRadius(query, r)) {
    if (space_->distance(query, states[id]) <= r) out.push_back(id);
  }
  return out;
}

}  // namespace rrt
