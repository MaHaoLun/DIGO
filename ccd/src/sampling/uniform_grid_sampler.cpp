#include "corpus/sampling.hpp"
#include "corpus/graph.hpp"

#include <algorithm>

namespace corpus {
std::vector<std::size_t> uniform_representatives(const Graph& graph, const SamplingConfig& config) {
    sampling_detail::validate(graph, config);
    auto buckets = sampling_detail::coarse_buckets(graph);
    sampling_detail::Random random(config.seed);
    std::vector<std::size_t> representatives;
    for (auto& bucket : buckets) {
        if (bucket.empty()) continue;
        // A full permutation gives a uniform sample without replacement, and
        // its first item is uniform among that selected cell's target nodes.
        random.shuffle(bucket);
        const auto count = std::min(bucket.size(), static_cast<std::size_t>(config.candidates_per_grid));
        representatives.insert(representatives.end(), bucket.begin(), bucket.begin() + count);
    }
    return representatives;
}
} // namespace corpus
