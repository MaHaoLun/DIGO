#pragma once

#include "corpus/records.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <unordered_map>
#include <vector>

namespace corpus {

enum class RoutingPolicy { LegacySlots, CanonicalFifoLpf };
inline constexpr const char* legacy_policy_name = "legacy_bidirectional_last_snapshot_row_per_unordered_pair_v1";
inline constexpr const char* fifo_policy_name = "canonical_pair_periodic_fifo_lpf_v1";
struct GraphAudit {
    std::uint64_t source_varying_rows=0, canonical_pair_count=0;
    std::uint64_t canonical_varying_pairs_before=0, canonical_varying_pairs_after=0;
    std::uint64_t changed_pairs=0, changed_knots=0, sum_raise=0, max_raise=0;
    std::uint64_t raw_lpf_violations=0, repaired_continuous_violations=0, repaired_integer_violations=0;
    std::uint64_t reverse_rows_ignored=0, duplicate_rows=0;
};

struct Node {
    std::int64_t id = 0;
    std::int64_t x = 0, y = 0; // Longitude and latitude, integer microdegrees.
    int coarse_cell = 0, fine_cell = 0; // Row-major 32x32 and 64x64 grids.
    float longitude_condition = 0, latitude_condition = 0;
};

struct Edge {
    std::size_t to = 0;
    std::array<std::int64_t, 96> weights{}; // Positive integer seconds.
};

class Graph {
public:
    // Nodes and each adjacency list are ordered by original node ID.
    std::vector<Node> nodes;
    std::vector<std::vector<Edge>> adjacency;
    std::vector<int> component;
    BBox bbox;
    RoutingPolicy policy = RoutingPolicy::LegacySlots;
    GraphAudit audit;
    std::int64_t travel_time(const Edge&, std::int64_t arrival) const;
    std::uint64_t input_arc_count = 0;
    // Source rows whose 96 weights are not constant, before last-row overrides.
    std::uint64_t input_time_varying_arc_count = 0;
    // Effective directed adjacency FIFO failures, including midnight.
    // Legacy checks step jumps; LPF checks continuous segments (also exactly
    // equivalent to integer-ceil FIFO for integer 900-second anchor spacing).
    std::uint64_t fifo_integer_violations = 0;

    // Throws std::out_of_range when the original node ID is absent.
    std::size_t index(std::int64_t original_id) const;
    static Graph load(const std::filesystem::path& snapshot,
                      const std::filesystem::path& coordinates,
                      const BBox& bbox, RoutingPolicy policy = RoutingPolicy::LegacySlots);

private:
    std::unordered_map<std::int64_t, std::size_t> indices_;
};

} // namespace corpus
