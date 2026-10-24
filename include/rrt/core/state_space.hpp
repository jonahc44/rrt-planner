#pragma once

#include <cstddef>

#include "rrt/core/local_path.hpp"
#include "rrt/core/types.hpp"

namespace rrt {

/// Describes a vehicle's configuration space: how to sample it, measure
/// distance in it, and steer between two states while respecting the
/// vehicle's motion constraints.
///
/// To plug in a new vehicle model, subclass this and implement the pure
/// virtuals. Planners never look inside a State; they only go through this
/// interface.
class StateSpace {
 public:
  virtual ~StateSpace() = default;

  /// Number of components in a State.
  virtual std::size_t dimension() const = 0;

  /// Uniform random state inside the space's bounds.
  virtual State sampleUniform(Rng& rng) const = 0;

  /// Length of the optimal steering path from `from` to `to`, i.e.
  /// connect(from, to)->length() (infinity if no path exists). May be
  /// asymmetric (e.g. Dubins), so argument order matters.
  virtual double distance(const State& from, const State& to) const = 0;

  /// Exact steering: a feasible motion that starts at `from` and ends exactly
  /// at `to`, ignoring obstacles. Returns nullptr if no such motion exists.
  virtual LocalPathPtr connect(const State& from, const State& to) const = 0;

  /// Steer from `from` toward `toward`, travelling at most `max_length`.
  /// Default: connect() and truncate.
  virtual LocalPathPtr steer(const State& from, const State& toward,
                             double max_length) const;

  // ---- Nearest-neighbour support -----------------------------------------
  //
  // Neighbour queries run in a Euclidean embedding of the space, indexed by a
  // KD-tree. The embedding must be a lower bound on distance() in both
  // directions:
  //
  //     || embed(a) - embed(b) ||_2  <=  distance(a, b)
  //
  // which lets the neighbour structure return exact results: it searches the
  // embedding with a conservative radius and filters by distance(). For R^n
  // the embedding is the identity; for Dubins it is (x, y).

  virtual std::size_t embeddingDimension() const = 0;
  virtual void embed(const State& s, double* out) const = 0;

  // ---- RRT* radius support -----------------------------------------------

  /// Lebesgue measure (volume) of the space, used to size the RRT* rewiring
  /// ball.
  virtual double measure() const = 0;
};

}  // namespace rrt
