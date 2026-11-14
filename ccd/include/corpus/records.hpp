#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace corpus {
enum class Split : std::uint8_t { train = 0, validation = 1, base_test = 2, adaptation = 3, post_update_test = 4 };
inline const char* split_name(Split value) {
    switch (value) {
        case Split::train: return "train";
        case Split::validation: return "validation";
        case Split::base_test: return "base_test";
        case Split::adaptation: return "adaptation";
        case Split::post_update_test: return "post_update_test";
    }
    return "invalid";
}
struct QueryRecord {
    std::uint64_t query_id = 0;
    std::uint64_t interval_id = 0;
    std::uint64_t row_index = 0; // Stable zero-based index inside this split.
    std::string city;
    std::string snapshot_id;
    std::int64_t source = 0;
    std::int64_t destination = 0;
    std::int64_t departure = 0; // Seconds since start of the observation day.
    Split split = Split::train;
};
struct RouteRecord {
    QueryRecord query;
    std::vector<std::int64_t> node_ids;
    std::vector<std::int64_t> arrival_times; // Unwrapped seconds, same length as nodes.
    std::int64_t travel_time = 0;
    std::string status = "ok";
};
struct BBox {
    std::int64_t longitude_min = 0, longitude_max = 0;
    std::int64_t latitude_min = 0, latitude_max = 0;
};
} // namespace corpus
