#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "rrt/core/state_space.hpp"

namespace rrt {

/// Incremental nearest-neighbour index over tree vertices. Ids are assigned
/// densely in insertion order (0, 1, 2, ...). Results are exact with respect
/// to StateSpace::distance(), including for asymmetric metrics.
class NearestNeighbors {
 public:
  virtual ~NearestNeighbors() = default;

  /// Insert a state; returns its id.
  virtual std::size_t add(const State& s) = 0;
  virtual std::size_t size() const = 0;
  virtual void clear() = 0;

  /// argmin_i distance(state_i, query). Requires size() > 0.
  virtual std::size_t nearest(const State& query) const = 0;

  /// All ids i with distance(state_i, query) <= r  ("can reach query").
  virtual std::vector<std::size_t> nearTo(const State& query,
                                          double r) const = 0;

  /// All ids i with distance(query, state_i) <= r  ("reachable from query").
  virtual std::vector<std::size_t> nearFrom(const State& query,
                                            double r) const = 0;

  /// A superset of nearTo(query, r) and nearFrom(query, r), found without
  /// evaluating the exact metric. Callers that compute steering paths to
  /// each neighbour anyway can filter by path length themselves and skip a
  /// redundant distance() per candidate. Default: the exact union.
  virtual std::vector<std::size_t> candidates(const State& query,
                                              double r) const;
};

/// Linear scan. Reference implementation for testing.
class BruteForceNearestNeighbors final : public NearestNeighbors {
 public:
  explicit BruteForceNearestNeighbors(std::shared_ptr<const StateSpace> space)
      : space_(std::move(space)) {}

  std::size_t add(const State& s) override;
  std::size_t size() const override { return states_.size(); }
  void clear() override { states_.clear(); }
  std::size_t nearest(const State& query) const override;
  std::vector<std::size_t> nearTo(const State& query, double r) const override;
  std::vector<std::size_t> nearFrom(const State& query,
                                    double r) const override;

 private:
  std::shared_ptr<const StateSpace> space_;
  std::vector<State> states_;
};

/// nanoflann dynamic KD-tree over StateSpace::embed(). Queries search the
/// embedding with a radius that is guaranteed (by the embedding's lower-bound
/// property) to contain every true answer, then filter with the exact
/// metric.
class KdTreeNearestNeighbors final : public NearestNeighbors {
 public:
  explicit KdTreeNearestNeighbors(std::shared_ptr<const StateSpace> space);
  ~KdTreeNearestNeighbors() override;

  std::size_t add(const State& s) override;
  std::size_t size() const override;
  void clear() override;
  std::size_t nearest(const State& query) const override;
  std::vector<std::size_t> nearTo(const State& query, double r) const override;
  std::vector<std::size_t> nearFrom(const State& query,
                                    double r) const override;
  /// Ids whose embedding lies within r of the query's.
  std::vector<std::size_t> candidates(const State& query,
                                      double r) const override;

 private:
  /// Ids whose embedding lies within r (Euclidean) of the query's.
  std::vector<std::size_t> embeddedRadius(const State& query, double r) const;

  struct Impl;
  std::shared_ptr<const StateSpace> space_;
  std::unique_ptr<Impl> impl_;
};

}  // namespace rrt
