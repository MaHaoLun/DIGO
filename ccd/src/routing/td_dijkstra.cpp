#include "corpus/routing.hpp"

#include <algorithm>
#include <functional>
#include <limits>
#include <queue>
#include <stdexcept>
#include <tuple>

namespace corpus {
namespace {

std::int64_t traverse(const Graph& graph, std::int64_t arrival, const Edge& edge) {
    if (arrival < 0) throw std::logic_error("negative arrival in route traversal");
    const auto weight = graph.travel_time(edge, arrival);
    if (weight <= 0) throw std::logic_error("nonpositive graph weight in route traversal");
    if (arrival > std::numeric_limits<std::int64_t>::max() - weight)
        throw std::overflow_error("TD arrival exceeds signed 64-bit seconds");
    return arrival + weight;
}

} // namespace

TdDijkstra::TdDijkstra(const Graph& graph) : graph_(graph) {
    if (graph_.adjacency.size() != graph_.nodes.size())
        throw std::invalid_argument("graph adjacency and node registry sizes disagree");
}

void TdDijkstra::run(std::int64_t original_source, std::int64_t departure) {
    run_impl(original_source,departure,nullptr);
}
void TdDijkstra::run(std::int64_t source,std::int64_t departure,const std::vector<std::int64_t>& targets) {
    run_impl(source,departure,&targets);
}
void TdDijkstra::run_impl(std::int64_t original_source,std::int64_t departure,const std::vector<std::int64_t>* targets) {
    has_run_ = false;
    targeted_ = targets!=nullptr;
    if (departure < 0 || departure >= 86400)
        throw std::invalid_argument("departure must be in [0,86400) seconds");
    source_id_ = original_source;
    departure_ = departure;
    source_known_ = false;
    try {
        source_index_ = graph_.index(original_source);
        source_known_ = true;
    } catch (const std::out_of_range&) {
        has_run_ = true;
        return;
    }

    const auto count = graph_.nodes.size();
    // Negative sentinel preserves INT64_MAX as a valid reachable arrival.
    arrivals_.assign(count, -1);
    predecessors_.assign(count, count);
    settled_.assign(count,false);
    requested_.assign(count,false);
    std::size_t remaining=0;
    if(targets) for(auto id:*targets) {
        try {const auto i=graph_.index(id); if(!requested_[i]) {requested_[i]=true; ++remaining;}}
        catch(const std::out_of_range&) {}
    }
    // Original IDs are explicit in heap keys for deterministic equal arrivals.
    using State = std::tuple<std::int64_t, std::int64_t, std::size_t>;
    std::priority_queue<State, std::vector<State>, std::greater<State>> heap;
    arrivals_[source_index_] = departure;
    heap.emplace(departure, source_id_, source_index_);
    while (!heap.empty()) {
        const auto [arrival, original_id, current] = heap.top();
        heap.pop();
        (void)original_id;
        if (settled_[current] || arrival != arrivals_[current]) continue;
        settled_[current] = true;
        if(targeted_ && requested_[current] && --remaining==0) break;
        for (const auto& edge : graph_.adjacency[current]) {
            if (edge.to >= count) throw std::logic_error("edge references an invalid internal node ID");
            // Strictly positive costs ensure a later-settled node cannot
            // improve an earlier-settled node, even on non-FIFO input.
            if (settled_[edge.to]) continue;
            const auto candidate = traverse(graph_, arrival, edge);
            if (arrivals_[edge.to] < 0 || candidate < arrivals_[edge.to]) {
                arrivals_[edge.to] = candidate;
                predecessors_[edge.to] = current;
                heap.emplace(candidate, graph_.nodes[edge.to].id, edge.to);
            }
        }
    }
    has_run_ = true;
}

RouteRecord TdDijkstra::route(const QueryRecord& query) const {
    if (!has_run_) throw std::logic_error("route requires a completed TD Dijkstra run");
    if (query.source != source_id_ || query.departure != departure_)
        throw std::invalid_argument("query source/departure does not match the completed SSSP run");
    RouteRecord record;
    record.query = query;
    if (!source_known_) {
        record.status = "unknown_source";
        return record;
    }
    std::size_t destination;
    try {
        destination = graph_.index(query.destination);
    } catch (const std::out_of_range&) {
        record.status = "unknown_destination";
        return record;
    }
    if(targeted_ && !requested_[destination]) throw std::invalid_argument("destination was not requested for target-limited SSSP");
    if (arrivals_[destination] < 0) {
        record.status = "unreachable";
        return record;
    }
    if(!settled_[destination]) throw std::logic_error("destination is not settled");
    const auto count = graph_.nodes.size();
    auto cursor = destination;
    while (true) {
        if (cursor >= count || record.node_ids.size() >= count)
            throw std::logic_error("broken or cyclic TD predecessor chain");
        record.node_ids.push_back(graph_.nodes[cursor].id);
        record.arrival_times.push_back(arrivals_[cursor]);
        if (cursor == source_index_) break;
        cursor = predecessors_[cursor];
    }
    std::reverse(record.node_ids.begin(), record.node_ids.end());
    std::reverse(record.arrival_times.begin(), record.arrival_times.end());
    if (record.node_ids.front() != query.source || record.node_ids.back() != query.destination ||
        record.arrival_times.front() != query.departure)
        throw std::logic_error("route endpoints or departure do not match query");
    for (std::size_t i = 1; i < record.node_ids.size(); ++i) {
        const auto previous = graph_.index(record.node_ids[i - 1]);
        const auto next = graph_.index(record.node_ids[i]);
        const auto& neighbors = graph_.adjacency[previous];
        const auto edge = std::lower_bound(neighbors.begin(), neighbors.end(), next,
            [](const Edge& candidate, std::size_t index) { return candidate.to < index; });
        if (edge == neighbors.end() || edge->to != next)
            throw std::logic_error("route contains a nonexistent edge");
        const auto expected = traverse(graph_, record.arrival_times[i - 1], *edge);
        if (record.arrival_times[i] != expected)
            throw std::logic_error("route arrival sequence disagrees with edge traversal");
    }
    record.travel_time = record.arrival_times.back() - query.departure;
    if (record.travel_time < 0 || record.travel_time != arrivals_[destination] - departure_)
        throw std::logic_error("route travel time disagrees with arrival labels");
    record.status = "ok";
    return record;
}

} // namespace corpus
