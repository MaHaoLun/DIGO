#pragma once

#include "corpus/graph.hpp"
#include "corpus/records.hpp"

#include <array>

namespace corpus {

struct EncodedRow {
    std::array<float, 2048> image;
    std::array<float, 5> odt;
    float arr = 0;
};

struct EncodingMetrics {
    double mapping_seconds = 0;
    double rasterize_seconds = 0;
    double encode_seconds = 0;
};

// Throws on malformed routes; every successful call represents exactly one row.
EncodedRow encode_route(const Graph& graph, const RouteRecord& route,
                        EncodingMetrics* metrics = nullptr);

} // namespace corpus
