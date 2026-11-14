#pragma once
#include <array>
#include <cstdint>
namespace corpus::lpf {
using Knots = std::array<std::int64_t,96>;
// Narrow adaptation of GES LPFunction getY/computeY: periodic linear knots,
// exact wide arithmetic and ceiling replace legacy integer division/Round.
std::int64_t evaluate(const Knots&, std::int64_t time);
Knots least_fifo_majorant(const Knots&);
std::uint64_t violations(const Knots&);
}
