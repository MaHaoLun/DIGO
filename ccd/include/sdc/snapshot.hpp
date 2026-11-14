#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace sdc {

using NodeId = std::uint64_t;
using RoadId = std::uint64_t;
using Weight = std::uint32_t;
inline constexpr std::size_t kSlotCount = 96;
inline constexpr std::int64_t kSlotSeconds = 900;
inline constexpr std::int64_t kDaySeconds = 86400;
inline constexpr const char* kAggregationPolicy = "legacy_running_timestamp_mean_top5_trim_slot_integer_mean_change_threshold_v1";

struct DayWindow {
    std::string observation_date;
    std::int64_t start_epoch = 0;
    std::int64_t end_epoch = 0;
};
// This stage deliberately permits only G0 and reserved G1 dates, UTC+08:00.
DayWindow day_window(const std::string& observation_date);

struct CityInputs {
    std::filesystem::path base_graph;
    std::filesystem::path coordinates;
    std::filesystem::path road_registry;
};
struct Coordinate {
    NodeId node_id = 0;
    std::int64_t longitude_microdegrees = 0;
    std::int64_t latitude_microdegrees = 0;
};
struct Edge {
    NodeId src = 0;
    NodeId dst = 0;
    Weight base_seconds = 0;
};
struct Road {
    RoadId road_id = 0;
    NodeId original_src = 0;
    NodeId original_dst = 0;
    std::uint64_t distance_m = 0;
    std::size_t edge_index = 0; // Existing canonical (min ID, max ID) base arc.
    Weight legacy_base_seconds = 0; // Last matching direction in the base graph.
};
struct Graph {
    std::size_t node_count = 0;
    // File order is retained. Road order is the historical edge-index order.
    std::vector<Coordinate> coordinates;
    std::vector<Edge> edges;
    std::vector<Road> roads;
    std::unordered_map<RoadId, std::size_t> road_index;
    std::uint64_t multiple_roads_same_arc = 0;
};
Graph load_city_inputs(const CityInputs& inputs);

struct Statistics {
    // Missing keys mean zero. Counters are deterministic and checked for overflow.
    std::map<std::string, std::uint64_t> counters;
    std::array<std::uint64_t, kSlotCount> slot_observations{};
    std::array<std::uint64_t, kSlotCount> slot_timestamp_samples{};
    std::array<std::uint64_t, kSlotCount> slot_updates{};
    std::int64_t min_selected_epoch = -1;
    std::int64_t max_selected_epoch = -1;
};
void increment_counter(Statistics& stats, const std::string& name,
                       std::uint64_t increment = 1);

struct TimestampMean {
    Weight seconds = 0;
    std::uint64_t count = 0;
};
struct RawAggregation {
    // key = road_index * 86400 + seconds since local midnight. No trajectories retained.
    std::unordered_map<std::uint64_t, TimestampMean> timestamp_means;
    Statistics stats;
};
// File order, line order, then road order are semantically significant.
// Structural CSV or static-input errors throw; anomalous reversed durations are counted.
RawAggregation accumulate_raw_trajectories(
    const Graph& graph, const std::vector<std::filesystem::path>& ordered_raw_paths,
    const DayWindow& day);

struct AggregationResult {
    // Only accepted legacy updates. An absent edge/slot uses its original base weight.
    std::array<std::map<std::size_t, Weight>, kSlotCount> updates;
    Statistics stats;
};
AggregationResult finalize_legacy_updates(const Graph& graph, RawAggregation raw,
                                          const DayWindow& day);
Weight snapshot_weight(const Graph& graph, const AggregationResult& result,
                       std::size_t edge_index, std::size_t slot);

} // namespace sdc
