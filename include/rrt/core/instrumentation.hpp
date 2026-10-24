#pragma once

// Decorators that count calls into a StateSpace or CollisionChecker. They
// wrap any implementation, so planners and spaces need no instrumentation of
// their own. Used by the performance-regression suite to measure work
// (steering solves, collision checks) independently of machine speed.

#include <atomic>
#include <cstdint>
#include <memory>

#include "rrt/core/collision_checker.hpp"
#include "rrt/core/state_space.hpp"

namespace rrt {

class CountingStateSpace final : public StateSpace {
 public:
  explicit CountingStateSpace(std::shared_ptr<const StateSpace> inner)
      : inner_(std::move(inner)) {}

  std::size_t dimension() const override { return inner_->dimension(); }
  State sampleUniform(Rng& rng) const override {
    samples_.fetch_add(1, std::memory_order_relaxed);
    return inner_->sampleUniform(rng);
  }
  double distance(const State& a, const State& b) const override {
    distances_.fetch_add(1, std::memory_order_relaxed);
    return inner_->distance(a, b);
  }
  LocalPathPtr connect(const State& a, const State& b) const override {
    connects_.fetch_add(1, std::memory_order_relaxed);
    return inner_->connect(a, b);
  }
  /// Forwards to the inner steer(); connect() calls it makes internally are
  /// not counted as connects.
  LocalPathPtr steer(const State& a, const State& b,
                     double max_length) const override {
    steers_.fetch_add(1, std::memory_order_relaxed);
    return inner_->steer(a, b, max_length);
  }
  std::size_t embeddingDimension() const override {
    return inner_->embeddingDimension();
  }
  void embed(const State& s, double* out) const override {
    inner_->embed(s, out);
  }
  double measure() const override { return inner_->measure(); }

  std::uint64_t sampleCalls() const { return samples_.load(); }
  std::uint64_t distanceCalls() const { return distances_.load(); }
  std::uint64_t connectCalls() const { return connects_.load(); }
  std::uint64_t steerCalls() const { return steers_.load(); }
  void reset() {
    samples_ = distances_ = connects_ = steers_ = 0;
  }

 private:
  std::shared_ptr<const StateSpace> inner_;
  mutable std::atomic<std::uint64_t> samples_{0}, distances_{0}, connects_{0},
      steers_{0};
};

class CountingCollisionChecker final : public CollisionChecker {
 public:
  explicit CountingCollisionChecker(std::shared_ptr<const CollisionChecker> inner)
      : CollisionChecker(inner->resolution()), inner_(std::move(inner)) {}

  bool isStateValid(const State& s) const override {
    states_.fetch_add(1, std::memory_order_relaxed);
    return inner_->isStateValid(s);
  }
  /// Forwards to the inner checker (keeping any specialised motion check).
  /// State checks it performs internally are not counted; checkedLength()
  /// measures that work instead.
  bool isMotionValid(const LocalPath& path) const override {
    motions_.fetch_add(1, std::memory_order_relaxed);
    const double len = path.length();
    // Accumulate in micro-units so the counter can stay an atomic integer.
    length_um_.fetch_add(static_cast<std::uint64_t>(len * 1e6),
                         std::memory_order_relaxed);
    return inner_->isMotionValid(path);
  }

  std::uint64_t stateCalls() const { return states_.load(); }
  std::uint64_t motionCalls() const { return motions_.load(); }
  /// Total arc length passed to isMotionValid().
  double checkedLength() const { return static_cast<double>(length_um_.load()) * 1e-6; }
  void reset() { states_ = motions_ = length_um_ = 0; }

 private:
  std::shared_ptr<const CollisionChecker> inner_;
  mutable std::atomic<std::uint64_t> states_{0}, motions_{0}, length_um_{0};
};

}  // namespace rrt
