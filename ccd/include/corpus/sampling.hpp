#pragma once

#include "corpus/records.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace corpus {
class Graph;

struct SamplingConfig {
    std::string method = "hierarchical_grid";
    std::uint64_t seed = 20260905;
    int cluster_count = 4;
    int candidates_per_grid = 4;
    int fine_grid_dim = 2;
    int min_distance = 2;
    int max_distance = 4;
    std::int64_t start_second = 28800;
    std::int64_t interval_seconds = 3600;
    std::int64_t point_step_seconds = 900;
    // Validates the complete half-open window and returns its point count.
    std::uint64_t points_per_group() const;
    std::array<std::uint64_t, 3> split_rows{800000, 100000, 100000};
};

struct QueryPlan {
    std::vector<QueryRecord> queries;
    std::uint64_t representative_count = 0;
    std::uint64_t candidate_groups = 0;
    std::uint64_t reachable_groups = 0;
};

std::vector<std::size_t> hierarchical_representatives(const Graph&, const SamplingConfig&);
// Returns all sampled targets by coarse cell; each cell's first item is its
// uniformly selected source representative. Original node indices are retained.
std::vector<std::size_t> uniform_representatives(const Graph&, const SamplingConfig&);
std::vector<std::size_t> cluster_representatives(const Graph&, const SamplingConfig&);
QueryPlan make_query_plan(const Graph&, const SamplingConfig&,
                         const std::string& city, const std::string& snapshot_id);

namespace sampling_detail {
// mt19937_64 output is standardized; this rejection mapping and Fisher-Yates
// order avoid implementation-specific uniform_int_distribution/std::shuffle.
class Random {
public:
    explicit Random(std::uint64_t seed) : engine_(seed) {}
    std::uint64_t bounded(std::uint64_t exclusive_upper);
    void shuffle(std::vector<std::size_t>& values);
private:
    std::mt19937_64 engine_;
};
void validate(const Graph&, const SamplingConfig&);
std::vector<std::vector<std::size_t>> coarse_buckets(const Graph&);
} // namespace sampling_detail
} // namespace corpus
