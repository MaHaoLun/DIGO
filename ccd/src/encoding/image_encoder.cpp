#include "corpus/encoding.hpp"

#include "sdc/io.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>

namespace corpus {
namespace {
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double>(end - start).count();
}
[[noreturn]] void invalid(const std::string& reason) {
    throw sdc::io::InputError("route encoding: " + reason);
}
} // namespace

EncodedRow encode_route(const Graph& graph, const RouteRecord& route,
                        EncodingMetrics* metrics) {
    const auto begin = metrics ? Clock::now() : Clock::time_point{};
    if (route.status != "ok") invalid("status must be ok");
    if (route.node_ids.empty() || route.node_ids.size() != route.arrival_times.size())
        invalid("nodes and arrivals must have equal nonzero lengths");
    if (route.query.departure < 0 || route.query.departure >= 86400)
        invalid("departure must be in [0,86400)");
    if (route.travel_time < 0) invalid("negative travel_time");
    if (route.node_ids.front() != route.query.source ||
        route.node_ids.back() != route.query.destination)
        invalid("route endpoints do not match query");
    if (route.arrival_times.front() != route.query.departure)
        invalid("first arrival differs from departure");
    if (route.arrival_times.back() < route.query.departure ||
        route.arrival_times.back() - route.query.departure != route.travel_time)
        invalid("final arrival and travel_time disagree");
    if (route.travel_time == 0 && route.node_ids.size() != 1)
        invalid("zero-duration route must contain exactly one node");

    EncodedRow result;
    result.image.fill(-1.0F);
    const Node* origin = nullptr;
    const Node* destination = nullptr;
    auto previous_arrival = route.query.departure;
    const double denominator = static_cast<double>(std::max<std::int64_t>(route.travel_time, 1));
    const auto initialized = metrics ? Clock::now() : Clock::time_point{};
    if (metrics) metrics->encode_seconds += elapsed(begin, initialized);
    for (std::size_t i = 0; i < route.node_ids.size(); ++i) {
        const auto mapping_begin = metrics ? Clock::now() : Clock::time_point{};
        const auto& node = graph.nodes.at(graph.index(route.node_ids[i]));
        if (node.coarse_cell < 0 || node.coarse_cell >= 1024)
            invalid("node has invalid fixed-grid mapping");
        if (!std::isfinite(node.longitude_condition) || !std::isfinite(node.latitude_condition) ||
            std::abs(node.longitude_condition) > 1 || std::abs(node.latitude_condition) > 1)
            invalid("node has invalid coordinate condition");
        if (i == 0) origin = &node;
        destination = &node;
        const auto mapped = metrics ? Clock::now() : Clock::time_point{};
        const auto arrival = route.arrival_times[i];
        if ((i > 0 && arrival <= previous_arrival) || arrival > route.arrival_times.back())
            invalid("arrival sequence must strictly increase within route cost");
        previous_arrival = arrival;
        const auto cell = static_cast<std::size_t>(node.coarse_cell);
        result.image[cell] = 1.0F;
        result.image[1024 + cell] = static_cast<float>(
            2.0 * static_cast<double>(arrival - route.query.departure) / denominator - 1.0);
        if (metrics) {
            const auto rasterized = Clock::now();
            metrics->mapping_seconds += elapsed(mapping_begin, mapped);
            metrics->rasterize_seconds += elapsed(mapped, rasterized);
        }
    }
    const auto condition_begin = metrics ? Clock::now() : Clock::time_point{};
    result.odt = {origin->longitude_condition, origin->latitude_condition,
                  destination->longitude_condition, destination->latitude_condition,
                  static_cast<float>(2.0 * static_cast<double>(route.query.departure) / 86400.0 - 1.0)};
    result.arr = static_cast<float>(static_cast<double>(route.travel_time) / 60.0);
    if (metrics) metrics->encode_seconds += elapsed(condition_begin, Clock::now());
    return result;
}

} // namespace corpus
