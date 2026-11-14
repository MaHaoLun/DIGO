#include "sdc/snapshot.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace sdc {
namespace {

struct SlotSum {
    std::uint64_t sum = 0;
    std::uint64_t count = 0;
};

void add(SlotSum& accumulator, Weight seconds) {
    if (accumulator.sum > std::numeric_limits<std::uint64_t>::max() - seconds ||
        accumulator.count == std::numeric_limits<std::uint64_t>::max())
        throw std::overflow_error("legacy slot aggregation overflow");
    accumulator.sum += seconds;
    ++accumulator.count;
}

bool legacy_threshold(Weight candidate, Weight previous) {
    const double change = static_cast<double>(candidate) - previous;
    if (change > 0) return change > 20 || change / previous > 1;
    // Deliberately preserve the source's signed test: change > 20 is false here.
    if (change < 0) return change > 20 || -change / previous > 0.5;
    return false;
}

} // namespace

AggregationResult finalize_legacy_updates(const Graph& graph, RawAggregation raw,
                                          const DayWindow& day) {
    const auto expected = day_window(day.observation_date);
    if (day.start_epoch != expected.start_epoch || day.end_epoch != expected.end_epoch)
        throw std::runtime_error("aggregation day window does not match observation date");
    AggregationResult result;
    result.stats = std::move(raw.stats);
    for (const char* name : {"trimmed_timestamp_samples", "retained_timestamp_samples", "observed_road_slots",
         "threshold_accepted_road_slots", "threshold_suppressed_road_slots", "duplicate_arc_slot_updates",
         "accepted_edge_slots", "observed_edge_slots", "roads_observed"})
        result.stats.counters[name] = 0;

    // Only city road/timestamp means are retained; complete trajectories are never buffered.
    using Sample = std::pair<Weight, std::uint32_t>;
    std::vector<std::vector<Sample>> road_samples(graph.roads.size());
    for (const auto& item : raw.timestamp_means) {
        const auto road_index = static_cast<std::size_t>(item.first / kDaySeconds);
        const auto offset = static_cast<std::uint32_t>(item.first % kDaySeconds);
        if (road_index >= graph.roads.size() || item.second.count == 0 || item.second.seconds == 0)
            throw std::runtime_error("invalid road/timestamp accumulator");
        road_samples[road_index].emplace_back(item.second.seconds, offset);
    }
    raw.timestamp_means.clear();
    raw.timestamp_means.rehash(0);

    using RoadSlot = std::pair<std::size_t, Weight>;
    std::array<std::vector<RoadSlot>, kSlotCount> candidates;
    std::array<std::map<std::size_t, bool>, kSlotCount> observed_edges;
    for (std::size_t road_index = 0; road_index < road_samples.size(); ++road_index) {
        auto& samples = road_samples[road_index];
        if (samples.empty()) continue;
        increment_counter(result.stats, "roads_observed");
        // Reference sorts pair(weight,timestamp), then removes ranked top 5% if n > 3.
        std::sort(samples.begin(), samples.end());
        const auto keep = samples.size() <= 3 ? samples.size() : samples.size() * 95 / 100;
        increment_counter(result.stats, "trimmed_timestamp_samples", samples.size() - keep);
        increment_counter(result.stats, "retained_timestamp_samples", keep);
        std::array<SlotSum, kSlotCount> slots{};
        for (std::size_t i = 0; i < samples.size(); ++i) {
            const auto slot = static_cast<std::size_t>(samples[i].second / kSlotSeconds);
            observed_edges[slot].emplace(graph.roads[road_index].edge_index, true);
            if (i < keep) {
                add(slots[slot], samples[i].first);
                if (result.stats.slot_timestamp_samples[slot] == std::numeric_limits<std::uint64_t>::max())
                    throw std::overflow_error("slot timestamp sample counter overflow");
                ++result.stats.slot_timestamp_samples[slot];
            }
        }
        for (std::size_t slot = 0; slot < kSlotCount; ++slot) {
            if (!slots[slot].count) continue;
            // The reference calls ceil AFTER integer division, thus retaining the floor.
            const auto weight = std::max<std::uint64_t>(1, slots[slot].sum / slots[slot].count);
            if (weight > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max()))
                throw std::overflow_error("legacy slot weight exceeds signed 32 bits");
            candidates[slot].emplace_back(road_index, static_cast<Weight>(weight));
            increment_counter(result.stats, "observed_road_slots");
        }
        std::vector<Sample>().swap(samples);
    }
    std::unordered_map<std::size_t, Weight> existing_updates;
    for (std::size_t slot = 0; slot < kSlotCount; ++slot) {
        increment_counter(result.stats, "observed_edge_slots", observed_edges[slot].size());
        for (const auto& candidate : candidates[slot]) {
            const auto edge_index = graph.roads[candidate.first].edge_index;
            if (edge_index >= graph.edges.size()) throw std::runtime_error("road edge index is invalid");
            const auto previous = existing_updates.find(edge_index);
            const auto old_weight = previous == existing_updates.end() ? graph.edges[edge_index].base_seconds : previous->second;
            if (legacy_threshold(candidate.second, old_weight)) {
                increment_counter(result.stats, "threshold_accepted_road_slots");
                // map::insert in the source preserves the FIRST accepted road for this arc/slot,
                // while its threshold state still changes on later accepted roads.
                if (!result.updates[slot].emplace(edge_index, candidate.second).second)
                    increment_counter(result.stats, "duplicate_arc_slot_updates");
                existing_updates[edge_index] = candidate.second;
            } else {
                increment_counter(result.stats, "threshold_suppressed_road_slots");
            }
        }
        result.stats.slot_updates[slot] = result.updates[slot].size();
        increment_counter(result.stats, "accepted_edge_slots", result.updates[slot].size());
    }
    return result;
}

Weight snapshot_weight(const Graph& graph, const AggregationResult& result,
                       std::size_t edge_index, std::size_t slot) {
    if (edge_index >= graph.edges.size() || slot >= kSlotCount)
        throw std::out_of_range("snapshot edge or slot index out of range");
    const auto update = result.updates[slot].find(edge_index);
    return update == result.updates[slot].end() ? graph.edges[edge_index].base_seconds : update->second;
}

} // namespace sdc
