#pragma once

#include "corpus/graph.hpp"

#include <cstdint>
#include <vector>

namespace corpus {

// One immediate-traversal TD Dijkstra run serves all matching destinations.
// Legacy non-FIFO input is accepted without an optimality certificate.
// Canonical FIFO LPF policy supports fixed-departure earliest arrival.
class TdDijkstra {
public:
    explicit TdDijkstra(const Graph& graph);
    // Departure must be in [0,86400). Arrival times remain unwrapped int64
    // seconds; Graph policy dispatches periodic slot or FIFO LPF evaluation.
    void run(std::int64_t original_source, std::int64_t departure);
    void run(std::int64_t original_source, std::int64_t departure,
             const std::vector<std::int64_t>& targets);
    // Throws on mismatched source/departure or a missing preceding run.
    // Unknown IDs and unreachable destinations return explicit statuses.
    RouteRecord route(const QueryRecord& query) const;

private:
    void run_impl(std::int64_t, std::int64_t, const std::vector<std::int64_t>*);
    std::vector<bool> settled_, requested_;
    bool targeted_ = false;
    const Graph& graph_;
    std::vector<std::int64_t> arrivals_;
    std::vector<std::size_t> predecessors_;
    std::int64_t source_id_ = 0, departure_ = 0;
    std::size_t source_index_ = 0;
    bool has_run_ = false, source_known_ = false;
};

} // namespace corpus
